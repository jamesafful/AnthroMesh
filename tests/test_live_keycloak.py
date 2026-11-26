"""Opt-in end-to-end token test using a disposable Keycloak realm and TLS proxy."""
import datetime as dt
import http.client
import os
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from anthromesh.adapters.heuristic import HeuristicAdapter
from anthromesh.cohort.oracle import ResearchCohortOracle
from anthromesh.protocols.mcp_server import MCPServer
from anthromesh.protocols.oauth import OAuthResource
from anthromesh.twin.engine import PersonalDecisionTwin
from anthromesh.vault.policy_engine import PolicyEngine
from anthromesh.vault.storage import BehavioralVault


def test_live_keycloak_scoped_mcp(tmp_path):
    upstream = os.getenv("ANTHROMESH_TEST_KEYCLOAK_PORT")
    if not upstream:
        pytest.skip("A disposable Keycloak test realm is required.")
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("jwt")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    class Proxy(BaseHTTPRequestHandler):
        def forward(self):
            length = int(self.headers.get("Content-Length", "0"))
            with_connection = http.client.HTTPConnection("127.0.0.1", int(upstream), timeout=10)
            try:
                with_connection.request(
                    self.command, self.path, body=self.rfile.read(length) if length else None,
                    headers={"Host": "localhost:18443",
                             "Content-Type": self.headers.get("Content-Type", "application/json")},
                )
                upstream_response = with_connection.getresponse()
                content = upstream_response.read()
                self.send_response(upstream_response.status)
                self.send_header("Content-Type", upstream_response.getheader("Content-Type"))
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            finally:
                with_connection.close()

        do_GET = do_POST = forward

        def log_message(self, *_args):
            pass

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(private_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), False)
            .sign(private_key, hashes.SHA256()))
    cert_file, key_file = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    server_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_tls.load_cert_chain(cert_file, key_file)
    with ThreadingHTTPServer(("127.0.0.1", 18443), Proxy) as proxy:
        proxy.socket = server_tls.wrap_socket(proxy.socket, server_side=True)
        worker = threading.Thread(target=proxy.serve_forever, daemon=True)
        worker.start()
        try:
            issuer = "https://localhost:18443/realms/anthromesh-test"
            oauth = OAuthResource(
                issuer, "https://anthromesh.example/mcp", discover=True, profile="keycloak",
                ca_file=cert_file, clock_skew_seconds=30,
            )
            ssl_context = ssl.create_default_context(cafile=str(cert_file))
            with httpx.Client(verify=ssl_context, trust_env=False) as client:
                response = client.post(issuer + "/protocol/openid-connect/token", data={
                    "grant_type": "client_credentials", "client_id": "smoke-client",
                    "client_secret": os.environ["ANTHROMESH_TEST_KEYCLOAK_SECRET"],
                })
                response.raise_for_status()
                bearer = "Bearer " + response.json()["access_token"]

            principal = oauth.authenticate(bearer)
            assert principal and principal.client_id == "smoke-client"
            vault = BehavioralVault(str(tmp_path / "test.db"))
            adapter = HeuristicAdapter()
            twin = PersonalDecisionTwin(vault, PolicyEngine(), adapter)
            oracle = ResearchCohortOracle(vault, adapter)
            with MCPServer(twin, oracle, oauth=oauth).create_http_server(port=0) as server:
                listener = threading.Thread(target=server.serve_forever, daemon=True)
                listener.start()
                try:
                    url = f"http://127.0.0.1:{server.server_port}/mcp"
                    headers = {"Mcp-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call",
                               "Mcp-Name": "twin_check_boundary"}
                    args = {"subject_id": principal.subject, "domain": "calendar", "context": "Check",
                            "candidate_options": [{"option_id": "a", "label": "A"},
                                                  {"option_id": "b", "label": "B"}]}
                    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "twin_check_boundary", "arguments": args,
                               "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                                         "io.modelcontextprotocol/clientCapabilities": {}}}}
                    with httpx.Client(trust_env=False) as client:
                        unauthorized = client.post(url, headers=headers, json=body)
                        assert unauthorized.status_code == 401
                        assert oauth.metadata_url in unauthorized.headers["www-authenticate"]
                        headers["Authorization"] = bearer
                        response = client.post(url, headers=headers, json=body)
                        assert response.json()["result"]["structuredContent"]["effect"] == "allow"
                        args["domain"] = "finance"
                        assert client.post(url, headers=headers, json=body).status_code == 403
                finally:
                    server.shutdown()
                    listener.join(timeout=2)
        finally:
            proxy.shutdown()
            worker.join(timeout=2)
