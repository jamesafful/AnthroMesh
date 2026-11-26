import hmac
import secrets
from html import escape
from typing import Callable
from urllib.parse import parse_qs, quote

from anthromesh.judgment.elicitation import Elicitation
from anthromesh.judgment.panel import HumanPanel, PanelAnswer
from anthromesh.models import EscalationStatus
from anthromesh.vault.storage import BehavioralVault


def create_inbox_app(
    vault: BehavioralVault, resolve: Callable[[str, str, str], dict], password: str,
    *, subject: str | None = None, panel_topic: str | None = None,
):
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from starlette.concurrency import run_in_threadpool
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, RedirectResponse

    if panel_topic and not subject:
        raise ValueError("Bind the panel inbox to a verified subject.")
    if not password:
        raise ValueError("Configure an owner inbox password.")
    app = FastAPI(title="AnthroMesh Owner Inbox", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    basic = HTTPBasic()
    elicitation = Elicitation(vault)
    panel = HumanPanel(vault) if panel_topic else None
    csrf_key = secrets.token_bytes(32)

    def authenticate(credentials: HTTPBasicCredentials = Depends(basic)) -> None:
        if (credentials.username != "owner"
                or not hmac.compare_digest(credentials.password.encode("utf-8"), password.encode("utf-8"))):
            raise HTTPException(401, headers={"WWW-Authenticate": "Basic realm=AnthroMesh"})

    def csrf(ticket_id: str) -> str:
        return hmac.digest(csrf_key, ticket_id.encode("utf-8"), "sha256").hex()

    @app.get("/", dependencies=[Depends(authenticate)])
    def inbox() -> HTMLResponse:
        items = []
        for ticket in vault.list_escalations(EscalationStatus.PENDING):
            if subject is not None and ticket.subject_id != subject:
                continue
            ticket_id = quote(ticket.ticket_id, safe="")
            options = "".join(f'<option value="{escape(item.option_id)}">{escape(item.label)}</option>'
                              for item in ticket.candidate_options)
            citations = []
            for identifier in ticket.evidence_ids:
                evidence = vault.get_decision(identifier)
                if evidence and evidence.subject_id == ticket.subject_id:
                    citations.append(f"<li>{escape(evidence.context)} ({escape(identifier)})</li>")
            references = (
                "<details><summary>Decision evidence</summary><ul>" + "".join(citations) + "</ul></details>"
                if citations else ""
            )
            items.append(f"""
            <article>
              <h2>{escape(ticket.domain)}</h2>
              <p>{escape(ticket.context)}</p><p>{escape(ticket.reason)}</p>
              {references}
              <form method="post" action="/resolve/{ticket_id}">
                <input type="hidden" name="csrf" value="{csrf(ticket.ticket_id)}">
                <label>Decision <select name="action" required>{options}</select></label>
                <label>Rationale <textarea name="rationale" required maxlength="4000"></textarea></label>
                <button type="submit">Resolve</button>
              </form>
            </article>""")
        if subject is not None:
            for case in elicitation.pending(subject):
                case_id = case["case_id"]
                options = "".join(
                    f'<option value="{escape(item["option_id"])}">{escape(item["label"])}</option>'
                    for item in case["candidate_options"]
                )
                condition = escape(case["changed_condition"] or "")
                items.append(f"""
                <article>
                  <h2>{escape(case["domain"])} scenario</h2>
                  <p>{escape(case["context"])}</p>
                  {f"<p>Changed condition: {condition}</p>" if condition else ""}
                  <form method="post" action="/respond/{quote(case_id, safe='')}">
                    <input type="hidden" name="csrf" value="{csrf('case:' + case_id)}">
                    <label>Decision <select name="action" required>{options}</select></label>
                    <label>Rationale <textarea name="rationale" maxlength="4000"></textarea></label>
                    <label>What would change your decision? <textarea name="switch_condition" maxlength="2000"></textarea></label>
                    <button type="submit">Submit response</button>
                  </form>
                </article>""")
        if panel is not None and subject is not None:
            assignment = panel.current(subject, panel_topic)
            if assignment is None:
                items.append(f"""
                <article><h2>Human panel</h2><form method="post" action="/panel/claim">
                  <input type="hidden" name="csrf" value="{csrf('panel:claim')}">
                  <input type="hidden" name="action" value="claim">
                  <button type="submit">Find a panel question</button>
                </form></article>""")
            else:
                key = assignment["assignment_id"]
                options = "".join(
                    f'<option value="{escape(option["option_id"])}">{escape(option["label"])}</option>'
                    for option in assignment["options"]
                )
                currency = escape(assignment["currency"])
                amount = assignment["payment_minor_if_approved"]
                items.append(f"""
                <article><h2>Human panel question</h2>
                <p>{escape(assignment["stimulus"])}</p>
                <p>Compensation after approval: {amount} minor units ({currency}).</p>
                <form method="post" action="/panel/answer/{quote(key, safe='')}">
                  <input type="hidden" name="csrf" value="{csrf('panel:' + key)}">
                  <label>Choice <select name="action" required>{options}</select></label>
                  <label>Rationale <textarea name="rationale" minlength="15" maxlength="5000" required></textarea></label>
                  <label><input type="checkbox" name="attest" value="yes" required> I authored this answer independently.</label>
                  <button type="submit">Submit panel answer</button>
                </form></article>""")
        content = "".join(items) if items else "<p>No pending escalations.</p>"
        return HTMLResponse(
            """<!doctype html><html lang="en"><meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>AnthroMesh Inbox</title><style>
            body{max-width:52rem;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui;color:#17242d}
            article{border:1px solid #b7c4cb;border-radius:.5rem;padding:1rem;margin-bottom:1rem}
            h2{margin-top:0} label{display:block;margin:.8rem 0} textarea{display:block;width:100%;box-sizing:border-box}
            input,button,select,textarea{font:inherit} button{padding:.4rem .85rem}
            </style><main><h1>Escalation inbox</h1>""" + content + "</main></html>",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'",
            },
        )

    async def parse_form(request: Request, identifier: str, max_fields: int) -> dict:
        if request.headers.get("content-type", "").split(";")[0] != "application/x-www-form-urlencoded":
            raise HTTPException(415, detail="Unsupported form encoding.")
        chunks = bytearray()
        async for chunk in request.stream():
            if len(chunks) + len(chunk) > 50000:
                raise HTTPException(413, detail="Form too large.")
            chunks.extend(chunk)
        try:
            fields = parse_qs(chunks.decode("utf-8"), keep_blank_values=True, max_num_fields=max_fields,
                              strict_parsing=True)
            form = {key: values[0] for key, values in fields.items() if len(values) == 1}
            state = form["csrf"]
        except (UnicodeError, ValueError, KeyError) as exc:
            raise HTTPException(400, detail="Invalid resolution form.") from exc
        if not state.isascii() or not hmac.compare_digest(state, csrf(identifier)):
            raise HTTPException(403, detail="Invalid resolution token.")
        if not form.get("action"):
            raise HTTPException(400, detail="Choose an action.")
        return form

    @app.post("/resolve/{ticket_id}", dependencies=[Depends(authenticate)])
    async def complete(ticket_id: str, request: Request) -> RedirectResponse:
        form = await parse_form(request, ticket_id, 3)
        if subject is not None:
            ticket = await run_in_threadpool(vault.get_escalation, ticket_id)
            if ticket is None or ticket.subject_id != subject:
                raise HTTPException(404, detail="Ticket not found.")
        action, rationale = form["action"], form.get("rationale", "")
        try:
            await run_in_threadpool(resolve, ticket_id, action, rationale)
        except KeyError as exc:
            raise HTTPException(404, detail="Ticket not found.") from exc
        except ValueError as exc:
            raise HTTPException(409, detail="Ticket could not be resolved.") from exc
        return RedirectResponse("/", status_code=303, headers={"Cache-Control": "no-store"})

    @app.post("/respond/{case_id}", dependencies=[Depends(authenticate)])
    async def respond(case_id: str, request: Request) -> RedirectResponse:
        if subject is None:
            raise HTTPException(404, detail="Scenario not found.")
        form = await parse_form(request, "case:" + case_id, 4)
        try:
            await run_in_threadpool(
                elicitation.respond, subject, case_id, form["action"], form.get("rationale", ""),
                form.get("switch_condition", ""),
            )
        except KeyError as exc:
            raise HTTPException(404, detail="Scenario not found.") from exc
        except ValueError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        return RedirectResponse("/", status_code=303, headers={"Cache-Control": "no-store"})

    @app.post("/panel/claim", dependencies=[Depends(authenticate)])
    async def claim_panel_question(request: Request) -> RedirectResponse:
        if panel is None or subject is None:
            raise HTTPException(404, detail="Panel unavailable.")
        await parse_form(request, "panel:claim", 2)
        try:
            assignment = await run_in_threadpool(panel.claim, subject, panel_topic)
        except ValueError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        if assignment is None:
            raise HTTPException(404, detail="No matching panel question is available.")
        return RedirectResponse("/", status_code=303, headers={"Cache-Control": "no-store"})

    @app.post("/panel/answer/{assignment_id}", dependencies=[Depends(authenticate)])
    async def answer_panel_question(assignment_id: str, request: Request) -> RedirectResponse:
        if panel is None or subject is None:
            raise HTTPException(404, detail="Panel unavailable.")
        form = await parse_form(request, "panel:" + assignment_id, 4)
        if form.get("attest") != "yes":
            raise HTTPException(400, detail="Independent authorship must be attested.")
        try:
            response = PanelAnswer(
                option_id=form["action"], rationale=form.get("rationale", ""),
                attests_independent_answer=True,
            )
            await run_in_threadpool(panel.answer, subject, assignment_id, response)
        except ValueError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        return RedirectResponse("/", status_code=303, headers={"Cache-Control": "no-store"})

    return app
