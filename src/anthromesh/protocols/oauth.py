from __future__ import annotations

import json
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Principal:
    subject: str
    client_id: str
    scopes: frozenset[str]

    def permits(self, name: str, arguments: dict[str, Any]) -> bool:
        if name in {"twin_query_decision", "twin_check_boundary", "twin_query_fast", "twin_trust_map"}:
            domain = arguments.get("domain")
            return (arguments.get("subject_id") == self.subject
                    and isinstance(domain, str) and f"twin:read:{domain}" in self.scopes)
        if name == "panel_read_result":
            topic = arguments.get("topic")
            return isinstance(topic, str) and f"panel:read:{topic}" in self.scopes
        if name == "cohort_simulate":
            cohort = arguments.get("cohort_id")
            return isinstance(cohort, str) and f"cohort:simulate:{cohort}" in self.scopes
        return False


class OAuthResource:
    """Verify access tokens with pinned issuer, audience, and trusted rotating public keys."""

    def __init__(
        self, issuer: str, resource: str, jwks_path: Path | None = None, *,
        discover: bool = False, profile: str = "rfc9068", ca_file: Path | None = None,
        transport: Any = None, clock_skew_seconds: int = 0,
    ) -> None:
        import jwt

        for url in (issuer, resource):
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("OAuth issuer and resource must be absolute HTTPS URLs.")
        if (jwks_path is None) == (not discover):
            raise ValueError("Provide exactly one of a JWKS file and live issuer discovery.")
        if profile not in {"rfc9068", "keycloak"}:
            raise ValueError("Unknown access token profile.")
        if not 0 <= clock_skew_seconds <= 30:
            raise ValueError("OAuth clock-skew allowance must be between 0 and 30 seconds.")
        self._clock_skew_seconds = clock_skew_seconds
        self._profile = profile
        self.issuer = issuer
        self.resource = resource
        self.host = urlsplit(resource).netloc
        self.metadata_url = f"https://{self.host}/.well-known/oauth-protected-resource/mcp"
        if urlsplit(resource).path != "/mcp":
            raise ValueError("OAuth resource URL must use the /mcp endpoint.")
        self.metadata = {
            "resource": resource,
            "authorization_servers": [issuer],
            "bearer_methods_supported": ["header"],
        }
        self._jwt = jwt
        self._remote_jwks = None
        self._lock = threading.Lock()
        self._refreshed = 0.0
        self._last_refresh_attempt = float("-inf")
        self._verify = ssl.create_default_context(cafile=str(ca_file)) if ca_file else True
        self._transport = transport
        if discover:
            discovery_url = issuer.rstrip("/") + "/.well-known/openid-configuration"
            metadata = self._fetch(discovery_url)
            if metadata.get("issuer") != issuer:
                raise ValueError("OIDC discovery issuer does not match the configured issuer.")
            self._remote_jwks = metadata.get("jwks_uri")
            if not isinstance(self._remote_jwks, str):
                raise ValueError("OIDC discovery did not provide a signing key URL.")
            parsed_keys = urlsplit(self._remote_jwks)
            issuer_host = urlsplit(issuer).netloc
            if (parsed_keys.scheme != "https" or parsed_keys.netloc != issuer_host
                    or parsed_keys.username or parsed_keys.password or parsed_keys.fragment
                    or parsed_keys.query):
                raise ValueError("The OIDC JWKS endpoint must use the issuer's HTTPS origin.")
            self.authorization_metadata = metadata
            self._keys = self._load_keys(self._fetch(self._remote_jwks))
            self._refreshed = self._last_refresh_attempt = time.monotonic()
        else:
            self.authorization_metadata = None
            self._keys = self._load_keys(json.loads(jwks_path.read_text(encoding="utf-8")))

    def _fetch(self, endpoint: str) -> dict:
        import httpx

        with httpx.Client(timeout=5, verify=self._verify, transport=self._transport,
                          follow_redirects=False, trust_env=False) as client:
            response = client.get(endpoint, headers={"Accept": "application/json"})
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("OAuth discovery returned an invalid document.")
            return payload

    def _load_keys(self, document: dict) -> dict:
        keys = {}
        for entry in document["keys"]:
            key_id = entry.get("kid")
            if (entry.get("kty") != "RSA" or entry.get("use", "sig") != "sig"
                    or entry.get("alg", "RS256") != "RS256"):
                continue
            if not isinstance(key_id, str) or not key_id or key_id in keys:
                raise ValueError("Trusted signing keys must have unique key IDs.")
            key = self._jwt.PyJWK.from_dict(entry, algorithm="RS256").key
            if key.key_size < 2048:
                raise ValueError("OAuth signing RSA keys must contain at least 2048 bits.")
            keys[key_id] = key
        if not keys:
            raise ValueError("OAuth JWKS has no RSA signing keys.")
        return keys

    def _signing_key(self, key_id: str):
        if self._remote_jwks is None:
            return self._keys.get(key_id)
        import httpx

        with self._lock:
            now = time.monotonic()
            age = now - self._refreshed
            key = self._keys.get(key_id)
            if key is not None and age < 300:
                return key
            if now - self._last_refresh_attempt < 10:
                return None
            self._last_refresh_attempt = now
            try:
                refreshed = self._load_keys(self._fetch(self._remote_jwks))
            except (httpx.HTTPError, ValueError, KeyError, TypeError, self._jwt.PyJWTError):
                return None
            self._keys = refreshed
            self._refreshed = now
            return self._keys.get(key_id)

    def authenticate(self, authorization: str) -> Principal | None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token or len(token) > 16384:
            return None
        try:
            header = self._jwt.get_unverified_header(token)
            keycloak = self._profile == "keycloak"
            types = {"jwt"} if keycloak else {"at+jwt", "application/at+jwt"}
            if header.get("typ", "").lower() not in types or header.get("alg") != "RS256":
                return None
            key_id = header.get("kid")
            if not isinstance(key_id, str) or not key_id:
                return None
            key = self._signing_key(key_id)
            if key is None:
                return None
            client_claim = "azp" if keycloak else "client_id"
            claims = self._jwt.decode(
                token, key, algorithms=["RS256"], issuer=self.issuer, audience=self.resource,
                leeway=self._clock_skew_seconds,
                options={"require": ["iss", "aud", "sub", client_claim, "iat", "exp", "jti"]},
            )
            if keycloak and claims.get("typ") != "Bearer":
                return None
            subject, client_id, scope = claims["sub"], claims[client_claim], claims.get("scope", "")
            if not all(isinstance(item, str) and item for item in (subject, client_id)):
                return None
            if not isinstance(scope, str):
                return None
            return Principal(subject, client_id, frozenset(scope.split()))
        except (self._jwt.PyJWTError, TypeError, AttributeError, ValueError):
            return None
