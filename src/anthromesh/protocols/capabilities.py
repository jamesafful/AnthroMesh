from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid

from anthromesh.vault.storage import BehavioralVault

from .oauth import Principal


class CapabilityTokens:
    """Issue opaque bearer capabilities bound to a subject, client, and exact scopes."""

    def __init__(self, vault: BehavioralVault) -> None:
        self.vault = vault

    def issue(
        self, subject: str, client: str, scopes: list[str], ttl_seconds: int, rate: int,
    ) -> tuple[str, str]:
        if not subject or not client or not scopes:
            raise ValueError("Subject, client, and at least one scope are required.")
        if not 1 <= ttl_seconds <= 86400 * 30 or not 1 <= rate <= 10000:
            raise ValueError("TTL must be one second to 30 days and rate one to 10000 per minute.")
        for scope in scopes:
            kind, sep, name = scope.rpartition(":")
            if kind not in {"twin:read", "cohort:simulate", "panel:read"} or not sep or not name:
                raise ValueError("Use twin:read:DOMAIN, cohort:simulate:COHORT, or panel:read:TOPIC scopes.")
            if "*" in name or any(character.isspace() for character in name):
                raise ValueError("Capability scopes require exact, whitespace-free names.")
        token_id = uuid.uuid4().hex
        secret = "am_" + secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(secret.encode()).hexdigest()
        self.vault.add_capability(
            token_id, token_hash, subject, client, sorted(set(scopes)),
            int(time.time()) + ttl_seconds, rate,
        )
        return token_id, secret

    def authenticate(self, authorization: str) -> Principal | None:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.startswith("am_") or len(token) != 46:
            return None
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        record = self.vault.consume_capability(token_hash, int(time.time()))
        if record is None:
            return None
        return Principal(
            record["subject_id"], record["client_id"], frozenset(json.loads(record["scopes_json"])),
        )
