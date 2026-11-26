# Delegation prior art and internal review

[OAuth token exchange](https://www.rfc-editor.org/rfc/rfc8693.html) defines
delegation and actor semantics. [Rich Authorization Requests](https://www.rfc-editor.org/rfc/rfc9396.html)
represents structured authority. [DPoP](https://www.rfc-editor.org/rfc/rfc9449.html)
sender-constrains tokens. [HTTP Message Signatures](https://www.rfc-editor.org/rfc/rfc9421.html)
signs selected message components. [GNAP](https://www.rfc-editor.org/rfc/rfc9635.html)
combines negotiated grants and key proofs. [Verifiable Credentials](https://www.w3.org/TR/vc-data-model-2.0/)
standardizes a model for issuer claims, holders, and verifiers.

AnthroMesh combines owner escalation, recording the exact proposed artifact,
and a relying-party audit entry in one workflow. That is a product distinction
to test with owners and relying parties. The signed grant, holder proof,
scope, and revocation components have established precedent.

Internal review covers exact artifact bytes through SHA-256, canonical
serialization of the typed intent, exact issuer and audience equality,
explicit scope and amount checks, and atomic one-time consumption of a
verifier-issued challenge. Verify and execute the exact bytes and typed action
in the relying party. Separate verifier deployments for the same audience
need a shared challenge store. A status statement can remain valid for up to
30 seconds after issue, so fetch it fresh at action time and plan around that
revocation bound. Servers require synchronized UTC clocks. Keep PostgreSQL
roles and schemas tenant-specific and test migrations and backup access
under those roles.

This is an internal review. An independent reviewer remains outstanding.
Ask them to inspect the actual relying-party action execution and tenant
provisioning in addition to the grant verifier.

For action binding, the closest product precedents are [AP2's transaction-bound mandates](https://github.com/google-agentic-commerce/AP2/blob/main/docs/ap2/agent_authorization.md) and the [Agentic Commerce Protocol's scoped payment tokens](https://agentic-commerce-protocol.com/docs/commerce/specs/payment). This review supports testing AnthroMesh's workflow for non-payment actions. It establishes no novelty claim for the authorization mechanism.
