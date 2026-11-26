import hashlib

import pytest

from anthromesh.judgment.panel import HumanPanel, PanelAnswer, PanelProfile, PanelQuestion, PersonAttestation
from anthromesh.models import CandidateOption
from anthromesh.protocols.delegation import (
    DelegationIssuer, GrantRequest, Intent, _public_string, _sign,
    issue_challenge, sign_action, verify_action,
)
from anthromesh.vault.privacy import SubjectData
from anthromesh.vault.storage import BehavioralVault


def test_delegation_is_holder_bound_revocable_scoped_and_one_use(tmp_path, monkeypatch):
    pytest.importorskip("jwt")
    ed25519 = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    issuer_vault = BehavioralVault(str(tmp_path / "issuer.db"))
    buyer = BehavioralVault(str(tmp_path / "buyer.db"))
    issuer_key, holder = (ed25519.Ed25519PrivateKey.generate() for _ in range(2))
    audience = "https://relying.example/approval"
    authority = DelegationIssuer(issuer_vault, "https://issuer.example", issuer_key)
    public = _public_string(issuer_key.public_key())
    request = GrantRequest(subject_id="owner", actor="agent",
                           audience=audience, operations=["commit"], domains=["calendar"],
                           resources=["reservation-1"], holder_public_key=_public_string(holder.public_key()))
    grant, token = authority.issue(request, now=1000)
    intent = Intent(operation="commit", domain="calendar", resource="reservation-1",
                    artifact_sha256=hashlib.sha256(b"reservation approved").hexdigest())
    status = authority.status(grant.jti, now=1000)
    challenge = issue_challenge(buyer, audience, intent, now=1000)
    proof = sign_action(token, public, holder, challenge, intent, now=1000)
    args = {"issuer": authority.issuer, "issuer_public_key": public, "audience": audience,
            "grant_token": token, "status_token": status, "proof_token": proof}
    with pytest.raises(ValueError, match="scope"):
        verify_action(buyer, **args, intent=intent.model_copy(update={"resource": "reservation-2"}), now=1001)
    with pytest.raises(ValueError, match="holder"):
        sign_action(token, public, issuer_key, challenge, intent, now=1001)
    observed_times = iter((1001, 1030))
    with monkeypatch.context() as patch:
        patch.setattr("anthromesh.protocols.delegation.time.time", lambda: next(observed_times))
        with pytest.raises(ValueError, match="at consumption"):
            verify_action(buyer, **args, intent=intent)
    assert verify_action(buyer, **args, intent=intent, now=1001)["actor"] == "agent"
    with pytest.raises(ValueError, match="already consumed"):
        verify_action(buyer, **args, intent=intent, now=1002)
    authority.revoke(grant.jti, now=1002)
    args["status_token"] = authority.status(grant.jti, now=1002)
    with pytest.raises(ValueError, match="revoked"):
        verify_action(buyer, **args, intent=intent, now=1003)
    assert buyer.verify_audit_chain() and issuer_vault.verify_audit_chain()


def test_panel_requires_identity_and_records_one_payment_per_approved_answer(tmp_path):
    pytest.importorskip("jwt")
    ed25519 = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    vault = BehavioralVault(str(tmp_path / "panel.db"))
    key = ed25519.Ed25519PrivateKey.generate()
    trust = {"issuer": "https://identity.example", "audience": "https://panel.example",
             "public_key": _public_string(key.public_key())}
    panel = HumanPanel(vault)

    def enroll(subject, identity):
        profile = PanelProfile(subject_id=subject, domains=["calendar"], expertise=["scheduling"],
                               agreed_to_answer=True, minimum_pay_minor=100, currency="USD")
        credential = _sign(PersonAttestation(
            iss=trust["issuer"], aud=trust["audience"], subject_id=subject, person_reference=identity,
            assurance="verified_human", iat=100, exp=100000,
        ), key, "anthromesh-person-attestation+jwt")
        panel.enroll(profile, credential, **trust, now=101)

    for index in range(3):
        enroll(f"person-{index}", f"unique-reference-{index}")
    with pytest.raises(ValueError, match="another participant"):
        enroll("duplicate", "unique-reference-1")
    question = PanelQuestion(question_id="calendar-1", topic="calendar", stimulus="Choose a time.",
                             options=[CandidateOption(option_id="a", label="Morning"),
                                      CandidateOption(option_id="b", label="Afternoon")],
                             expertise=["scheduling"], answer_count=3,
                             pay_per_approved_answer_minor=110, budget_minor=330, currency="USD")
    panel.submit_question(question, now=200)
    for index in range(3):
        subject = f"person-{index}"
        assignment = panel.claim(subject, "calendar", now=201)
        assert assignment == panel.claim(subject, "calendar", now=201)
        answer = PanelAnswer(option_id="a", rationale="Morning leaves a clear afternoon.",
                             attests_independent_answer=True)
        panel.answer(subject, assignment["assignment_id"], answer, now=202)
        with pytest.raises(ValueError, match="other than"):
            panel.review(assignment["assignment_id"], subject, True, "Independent answer", now=203)
        panel.review(assignment["assignment_id"], "moderator", True, "Clear and relevant", now=203)
        panel.review(assignment["assignment_id"], "moderator", True, "Clear and relevant", now=203)
    assert panel.report(question.question_id)["option_counts"] == {"a": 3, "b": 0}
    subject_data = SubjectData(vault)
    assert len(subject_data.export("person-0")["panel"]["compensation"]) == 1
    subject_data.erase("person-0")
    assert panel.report(question.question_id)["option_counts"] is None
    enroll("person-3", "unique-reference-3")
    assert panel.claim("person-3", "calendar", now=204) is None
