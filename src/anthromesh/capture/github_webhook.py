import hmac
import json

from anthromesh.vault.storage import BehavioralVault

from .workflows import github_reviews


def create_github_capture_app(
    vault: BehavioralVault, *, secret: str, subject: str, account: str, repositories: set[str],
):
    from fastapi import FastAPI, HTTPException
    from starlette.concurrency import run_in_threadpool
    from starlette.requests import Request
    from starlette.responses import Response

    if not secret or not subject or not account or not repositories:
        raise ValueError("Configure a secret, subject, GitHub account, and repository allowlist.")
    app = FastAPI(title="AnthroMesh GitHub capture", docs_url=None, redoc_url=None, openapi_url=None)
    key = secret.encode("utf-8")
    allowed = {item.casefold() for item in repositories}

    async def capture(request: Request):
        signatures = request.headers.getlist("x-hub-signature-256")
        if len(signatures) != 1 or not signatures[0].isascii():
            raise HTTPException(401, detail="Invalid delivery signature.")
        chunks = bytearray()
        async for chunk in request.stream():
            if len(chunks) + len(chunk) > 2 * 1024 * 1024:
                raise HTTPException(413, detail="Payload too large.")
            chunks.extend(chunk)
        expected = "sha256=" + hmac.digest(key, chunks, "sha256").hex()
        if not hmac.compare_digest(expected, signatures[0]):
            raise HTTPException(401, detail="Invalid delivery signature.")
        if request.headers.get("x-github-event") != "pull_request_review":
            return Response(status_code=204)

        try:
            payload = json.loads(chunks)
            repository = payload["repository"]["full_name"]
            if repository.casefold() not in allowed or payload["action"] != "submitted":
                return Response(status_code=204)
            decisions = github_reviews([payload], subject, account)
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise HTTPException(400, detail="Invalid review payload.") from exc

        count = await run_in_threadpool(vault.import_captured_decisions, decisions)
        return {"imported": count}

    app.add_api_route("/webhooks/github", capture, methods=["POST"])
    return app
