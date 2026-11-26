import json
import time

import pytest

from anthromesh.protocols.oauth import OAuthResource


def test_verified_oauth_principal_cannot_access_other_subjects_or_domains(tmp_path) -> None:
    jwt = pytest.importorskip("jwt")
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import rsa

    signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key(), as_dict=True)
    jwk["kid"] = "test-key"
    jwks = tmp_path / "keys.json"
    jwks.write_text(json.dumps({"keys": [jwk]}))
    issuer = "https://issuer.example"
    resource_url = "https://anthromesh.example/mcp"
    resource = OAuthResource(issuer, resource_url, jwks)
    now = int(time.time())
    claims = {
        "iss": issuer, "aud": resource_url, "sub": "owner", "client_id": "assistant",
        "iat": now, "exp": now + 60, "jti": "token-1", "scope": "twin:read:calendar",
    }

    def authorization(**changes) -> str:
        token = jwt.encode({**claims, **changes}, signing_key, algorithm="RS256",
                           headers={"kid": "test-key", "typ": "at+jwt"})
        return f"Bearer {token}"

    principal = resource.authenticate(authorization())
    assert principal.client_id == "assistant"
    assert principal.permits("twin_query_decision", {"subject_id": "owner", "domain": "calendar"})
    assert not principal.permits("twin_query_decision", {"subject_id": "other", "domain": "calendar"})
    assert not principal.permits("twin_query_decision", {"subject_id": "owner", "domain": "legal"})
    assert resource.authenticate(authorization(iss="https://different.example")) is None
    assert resource.authenticate(authorization(aud="https://wrong.example")) is None
    assert resource.authenticate(authorization(exp=now - 1)) is None
