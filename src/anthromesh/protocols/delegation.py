from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from anthromesh.vault.storage import BehavioralVault


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _https(value: str) -> str:
    target = urlsplit(value)
    if (target.scheme != "https" or not target.hostname or target.username is not None
            or target.password is not None or target.query or target.fragment
            or any(character.isspace() for character in value)):
        raise ValueError("Use a concrete HTTPS issuer or audience URL.")
    return value


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", value):
        raise ValueError("Invalid Ed25519 public key.")
    return base64.urlsafe_b64decode(value + "=")


def _public_key(raw: str):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = Ed25519PublicKey.from_public_bytes(_unb64(raw))
    if _public_string(key) != raw:
        raise ValueError("Use the canonical public key encoding.")
    return key


def _public_bytes(key) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _public_string(key) -> str:
    return _b64(_public_bytes(key))


def read_private_key(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Provide an Ed25519 private key.")
    return key


def create_private_key(path: Path) -> str:
    import os

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as output:
        output.write(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
    return _public_string(key.public_key())


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    operation: Literal["speak", "commit"]
    domain: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    resource: str = Field(min_length=1, max_length=300)
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    amount_minor: int | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")

    def digest(self) -> str:
        if (self.amount_minor is None) != (self.currency is None):
            raise ValueError("Specify amount and currency together.")
        return _hash(_canonical(self.model_dump()))


class GrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    subject_id: str = Field(min_length=1, max_length=128)
    actor: str = Field(min_length=1, max_length=128)
    audience: str
    operations: list[Literal["speak", "commit"]] = Field(min_length=1, max_length=2)
    domains: list[str] = Field(min_length=1, max_length=32)
    resources: list[str] = Field(min_length=1, max_length=64)
    holder_public_key: str
    ttl_seconds: int = Field(default=3600, ge=1, le=86400)
    max_amount_minor: int | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")

    @field_validator("audience")
    @classmethod
    def validate_audience(cls, value: str) -> str:
        return _https(value)

    def check(self) -> None:
        _public_key(self.holder_public_key)
        if (len(set(self.operations)) != len(self.operations)
                or len(set(self.domains)) != len(self.domains)
                or len(set(self.resources)) != len(self.resources)
                or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", item) for item in self.domains)
                or any(not item.strip() or "*" in item or len(item) > 300 for item in self.resources)):
            raise ValueError("Grant scopes require explicit, distinct domains, resources, and operations.")
        if (self.max_amount_minor is None) != (self.currency is None):
            raise ValueError("Specify a monetary limit and its currency together.")

    def permits(self, intent: Intent) -> bool:
        if intent.operation not in self.operations or intent.domain not in self.domains or intent.resource not in self.resources:
            return False
        if intent.amount_minor is None:
            return self.max_amount_minor is None
        return (self.max_amount_minor is not None and intent.currency == self.currency
                and intent.amount_minor <= self.max_amount_minor)


class Grant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    iss: str
    jti: str = Field(pattern=r"^[a-f0-9]{32}$")
    iat: int
    nbf: int
    exp: int
    authority: GrantRequest


class GrantStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    iss: str
    jti: str
    grant_sha256: str
    active: bool
    iat: int
    exp: int


class ActionProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    actor: str
    audience: str
    grant_id: str
    intent_sha256: str
    challenge: str
    iat: int
    exp: int


def _kid(public) -> str:
    return _hash(_public_bytes(public))[:24]


def _sign(value: BaseModel, key, token_type: str) -> str:
    import jwt

    return jwt.encode(value.model_dump(), key, algorithm="EdDSA",
                      headers={"kid": _kid(key.public_key()), "typ": token_type})


def _decode(token: str, key, token_type: str, model):
    import jwt

    if not isinstance(token, str) or len(token) > 16384:
        raise ValueError("Invalid signed statement.")
    try:
        header = jwt.get_unverified_header(token)
        if header != {"alg": "EdDSA", "kid": _kid(key), "typ": token_type}:
            raise ValueError("The statement has an unexpected signing key or type.")
        data = jwt.decode(token, key, algorithms=["EdDSA"], options={
            "verify_exp": False, "verify_iat": False, "verify_nbf": False,
            "verify_aud": False, "verify_iss": False, "verify_sub": False, "verify_jti": False,
        })
        return model.model_validate(data)
    except jwt.PyJWTError as exc:
        raise ValueError("Invalid signed statement.") from exc


class DelegationIssuer:
    def __init__(self, vault: BehavioralVault, issuer: str, private_key) -> None:
        self.vault = vault
        self.issuer = _https(issuer)
        self.key = private_key

    def issue(self, request: GrantRequest, *, now: int | None = None) -> tuple[Grant, str]:
        request.check()
        now = int(time.time()) if now is None else now
        grant = Grant(iss=self.issuer, jti=secrets.token_hex(16), iat=now, nbf=now,
                      exp=now + request.ttl_seconds, authority=request)
        token = _sign(grant, self.key, "anthromesh-delegation+jwt")
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.vault._assert_active_subject(conn, request.subject_id)
            conn.execute(
                """INSERT INTO delegation_grants(grant_id, subject_id, grant_hash, payload_json)
                   VALUES (?, ?, ?, ?)""",
                (grant.jti, request.subject_id, _hash(token.encode()), grant.model_dump_json()),
            )
            self.vault._append_audit(
                conn, "delegation_issued", "vault_owner", request.subject_id, "authority", "issued", None,
                {"grant_id": grant.jti, "actor": request.actor, "audience": request.audience,
                 "operations": request.operations, "expires_at": grant.exp},
            )
        return grant, token

    def revoke(self, grant_id: str, *, now: int | None = None) -> None:
        now = int(time.time()) if now is None else now
        with self.vault._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT subject_id, payload_json, revoked_at FROM delegation_grants WHERE grant_id = ?",
                (grant_id,),
            ).fetchone()
            if row is None or Grant.model_validate_json(row["payload_json"]).iss != self.issuer:
                raise KeyError(grant_id)
            if row["revoked_at"] is not None:
                return
            conn.execute("UPDATE delegation_grants SET revoked_at = ? WHERE grant_id = ?", (now, grant_id))
            self.vault._append_audit(
                conn, "delegation_revoked", "vault_owner", row["subject_id"], "authority", "revoked", None,
                {"grant_id": grant_id},
            )

    def status(self, grant_id: str, *, now: int | None = None) -> str:
        now = int(time.time()) if now is None else now
        with self.vault._connect() as conn:
            row = conn.execute(
                "SELECT grant_hash, payload_json, revoked_at FROM delegation_grants WHERE grant_id = ?",
                (grant_id,),
            ).fetchone()
        if row is None:
            raise KeyError(grant_id)
        grant = Grant.model_validate_json(row["payload_json"])
        if grant.iss != self.issuer:
            raise KeyError(grant_id)
        return _sign(GrantStatus(
            iss=self.issuer, jti=grant_id, grant_sha256=row["grant_hash"],
            active=row["revoked_at"] is None and grant.nbf <= now < grant.exp,
            iat=now, exp=now + 30,
        ), self.key, "anthromesh-delegation-status+jwt")

    def jwks(self) -> dict:
        public = self.key.public_key()
        return {"keys": [{"kty": "OKP", "crv": "Ed25519", "alg": "EdDSA",
                          "use": "sig", "kid": _kid(public), "x": _public_string(public)}]}


def issue_challenge(vault: BehavioralVault, audience: str, intent: Intent, *, now: int | None = None) -> str:
    now = int(time.time()) if now is None else now
    challenge = secrets.token_urlsafe(32)
    with vault._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM delegation_challenges WHERE expires_at <= ?", (now,))
        conn.execute("INSERT INTO delegation_challenges VALUES (?, ?, ?, ?, NULL)",
                     (challenge, _https(audience), intent.digest(), now + 90))
    return challenge


def sign_action(grant_token: str, issuer_public_key: str, holder_key, challenge: str,
                intent: Intent, *, now: int | None = None) -> str:
    issuer = _public_key(issuer_public_key)
    grant = _decode(grant_token, issuer, "anthromesh-delegation+jwt", Grant)
    now = int(time.time()) if now is None else now
    grant.authority.check()
    if (not grant.nbf <= now < grant.exp or not grant.authority.permits(intent)
            or _public_string(holder_key.public_key()) != grant.authority.holder_public_key
            or not re.fullmatch(r"[A-Za-z0-9_-]{43}", challenge)):
        raise ValueError("The delegation does not authorize this holder or action.")
    return _sign(ActionProof(
        actor=grant.authority.actor, audience=grant.authority.audience, grant_id=grant.jti,
        intent_sha256=intent.digest(), challenge=challenge, iat=now, exp=min(now + 60, grant.exp),
    ), holder_key, "anthromesh-delegated-action+jwt")


def verify_action(
    vault: BehavioralVault, *, issuer: str, issuer_public_key: str, audience: str,
    grant_token: str, status_token: str, proof_token: str, intent: Intent,
    now: int | None = None,
) -> dict:
    use_system_clock = now is None
    now = int(time.time()) if use_system_clock else now
    issuer, audience = _https(issuer), _https(audience)
    key = _public_key(issuer_public_key)
    grant = _decode(grant_token, key, "anthromesh-delegation+jwt", Grant)
    status = _decode(status_token, key, "anthromesh-delegation-status+jwt", GrantStatus)
    grant.authority.check()
    digest = intent.digest()
    if (grant.iss != issuer or status.iss != issuer or grant.authority.audience != audience
            or not grant.iat <= grant.nbf <= now < grant.exp
            or not 0 < grant.exp - grant.iat <= 86400
            or status.jti != grant.jti or status.grant_sha256 != _hash(grant_token.encode())
            or not status.active or not grant.nbf <= status.iat <= now < status.exp
            or not 0 < status.exp - status.iat <= 30 or not grant.authority.permits(intent)):
        raise ValueError("Delegation is invalid, expired, revoked, or out of scope.")
    holder = _public_key(grant.authority.holder_public_key)
    proof = _decode(proof_token, holder, "anthromesh-delegated-action+jwt", ActionProof)
    if (proof.actor != grant.authority.actor or proof.audience != audience
            or proof.grant_id != grant.jti or proof.intent_sha256 != digest
            or not grant.iat <= proof.iat <= now < proof.exp <= grant.exp
            or not 0 < proof.exp - proof.iat <= 60):
        raise ValueError("The holder's proof does not match the delegated action.")
    with vault._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if use_system_clock:
            now = int(time.time())
        if not max(grant.nbf, status.iat, proof.iat) <= now < min(grant.exp, status.exp, proof.exp):
            raise ValueError("The signed authorization is outside its validity period at consumption.")
        consumed = conn.execute(
            """UPDATE delegation_challenges SET consumed_at = ?
               WHERE challenge = ? AND audience = ? AND intent_hash = ? AND consumed_at IS NULL
                 AND expires_at > ?""", (now, proof.challenge, audience, digest, now),
        )
        if consumed.rowcount != 1:
            raise ValueError("The relying party challenge is unknown, expired, or already consumed.")
        vault._append_audit(
            conn, "delegation_verified", grant.authority.actor, grant.authority.subject_id,
            intent.domain, intent.operation, None,
            {"grant_id": grant.jti, "intent_sha256": digest, "status_sha256": _hash(status_token.encode())},
        )
    return {"authorized": True, "subject_id": grant.authority.subject_id,
            "actor": grant.authority.actor, "grant_id": grant.jti, "intent_sha256": digest,
            "operation": intent.operation}


def create_status_app(issuer: DelegationIssuer):
    from fastapi import FastAPI, HTTPException
    from starlette.responses import JSONResponse

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/.well-known/anthromesh-delegation-jwks.json")
    def keys():
        return JSONResponse(issuer.jwks(), headers={"Cache-Control": "no-store"})

    @app.get("/delegation/status/{grant_id}")
    def status(grant_id: str):
        if not re.fullmatch(r"[a-f0-9]{32}", grant_id):
            raise HTTPException(404, detail="Grant not found.")
        try:
            statement = issuer.status(grant_id)
        except KeyError as exc:
            raise HTTPException(404, detail="Grant not found.") from exc
        return JSONResponse({"status": statement}, headers={"Cache-Control": "no-store"})

    return app
