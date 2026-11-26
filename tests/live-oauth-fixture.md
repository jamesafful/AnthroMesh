# Reproduce the live OAuth smoke test

The realm export is for a disposable loopback-only test. Keycloak requires
the imported filename to end in `anthromesh-test-realm.json` because the
realm is named `anthromesh-test`. The test terminates HTTPS at
`https://localhost:18443`. Configure that hostname on Keycloak so discovery
advertises the same issuer the test verifies.

From the repository root in PowerShell:

```powershell
$realm = (Resolve-Path 'tests/anthromesh-test-realm.json').Path
docker run --rm -d --name anthromesh-oidc-smoke -e 'KC_HOSTNAME=https://localhost:18443' -p 127.0.0.1:18880:8080 --mount "type=bind,source=$realm,target=/opt/keycloak/data/import/anthromesh-test-realm.json,readonly" quay.io/keycloak/keycloak:26.4 start-dev --import-realm
$env:ANTHROMESH_TEST_KEYCLOAK_PORT = '18880'
$env:ANTHROMESH_TEST_KEYCLOAK_SECRET = 'anthromesh-disposable-secret'
```

Wait until the realm's discovery endpoint at
`http://127.0.0.1:18880/realms/anthromesh-test/.well-known/openid-configuration`
returns HTTP 200. Then run `python -m pytest tests/test_live_keycloak.py`.
Stop the fixture with `docker stop anthromesh-oidc-smoke`.
