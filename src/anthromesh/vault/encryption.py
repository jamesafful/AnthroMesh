from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .storage import BehavioralVault


def export_encrypted_vault(source: Path, destination: Path, passphrase: str) -> None:
    """Copy a plaintext vault to a new SQLCipher database and leave the source intact."""
    from sqlcipher3 import dbapi2

    if not passphrase or "\x00" in passphrase:
        raise ValueError("Provide a non-empty SQLCipher passphrase.")
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as stream:
        if stream.read(16) != b"SQLite format 3\x00":
            raise ValueError("The source must be an existing plaintext SQLite vault.")

    handle, temporary_name = tempfile.mkstemp(prefix=".anthromesh-", dir=destination.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        connection = dbapi2.connect(str(temporary), uri=True)
        try:
            if connection.execute("PRAGMA cipher_version").fetchone() is None:
                raise RuntimeError("The configured SQLite driver lacks SQLCipher support.")
            quoted = passphrase.replace("'", "''")
            connection.execute(f"PRAGMA key = '{quoted}'")
            connection.execute("ATTACH DATABASE ? AS plaintext KEY ''", (source.as_uri() + "?mode=ro",))
            tables = connection.execute(
                "SELECT name FROM plaintext.sqlite_master WHERE type = 'table'"
            ).fetchall()
            if not {"audit_logs", "decisions", "escalations", "cohorts"} <= {row[0] for row in tables}:
                raise ValueError("The source does not contain an AnthroMesh vault.")
            connection.execute("SELECT sqlcipher_export('main', 'plaintext')").fetchone()
            connection.execute("DETACH DATABASE plaintext")
            connection.commit()
        finally:
            connection.close()
        exported = BehavioralVault(str(temporary), encryption_key=passphrase)
        if not exported.verify_audit_chain():
            raise ValueError("Audit verification failed in the exported vault.")
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
