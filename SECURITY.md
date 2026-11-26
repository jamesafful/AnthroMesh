# Threat model and security reporting

Report a suspected vulnerability privately to the project's operator or maintainer.
Give the affected version, affected control, and sanitized observations. Avoid
including human histories, signing keys, bearer tokens, and replayable proofs.

Trusted boundaries are the authenticated operator and owner, installed
credentials, the vault and its database role, and the relying party that
verifies and executes a commitment. Query and retrieval text is data. It
does not confer authority. In memory-only mode, exact preferences may be
consulted and changed text, monetary fields, non-low risk, or structured query
metadata require owner review. Applications should put all material conditions
into the recorded context. Policy operates independently of remembered choices.

Use a separate PostgreSQL role and schema per tenant. An action verifier must
pin keys, consume each challenge transactionally, and execute only the bytes
and typed intent it verified. Signed status has a bounded lifetime. Check
it at action time and do not treat owner-memory provenance as a delegation.

The tests exercise replay, action scope and lifetime, concurrent challenge
consumption, structured-context changes, and database tenant boundaries.
These are internal checks. Independent review and actual relying-party
deployment constraints remain separate verification tasks.

## Independent review requested

We welcome review of the public delegation and verifier implementation.
The scope includes action and audience binding, concurrent challenge
consumption, revocation at use, expiry, and tenant boundaries. Use GitHub
Issues to express interest or report integration problems. Send suspected
vulnerabilities through the private reporting channel above.
