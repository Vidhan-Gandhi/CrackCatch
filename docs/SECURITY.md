# Security notes

**This is a research prototype. It is not hardened for internet exposure.** What follows is an
honest account of what exists, what does not, and what must be added before deployment.

---

## Authentication and authorisation

`AUTH_ENABLED=false` by default, so the demo and the tests need no setup. When enabled,
`backend/app/core/security.py` compares a static bearer token against a configured map and resolves
a role.

| Role | Maps to the scope's stakeholder group | Permissions |
|---|---|---|
| `authority` | Municipal corporations / road maintenance | Full: read, workflow, ingestion |
| `transport` | Traffic & transport departments | Read + analytics |
| `planner` | Smart-city infrastructure planners | Read + analytics |
| `citizen` | Drivers & commuters | Crowdsource submission only |

### What this is, and is not

**Is:** a demonstration that the system is *designed* around four distinct stakeholder roles — a
named scope requirement — with enforcement at the route level and tests proving a `citizen` token
cannot read the dashboard and a `transport` token cannot mutate a defect.

**Is not** production authentication. There is no user store, no password hashing, no token expiry,
no refresh, no revocation, no rate limiting, no audit of authentication attempts. Tokens are
long-lived strings in configuration.

Token comparison uses `hmac.compare_digest` against every configured token, so a timing side channel
cannot reveal which prefix was correct.

### Before deployment

Replace it with OIDC/OAuth2 against the municipality's identity provider. The role model maps
cleanly onto IdP groups; only `resolve_role` needs to change.

---

## Input handling

**Path traversal** — `POST /api/ingest/run` takes a server-side path, so it is sandboxed: the
resolved path must live inside the project directory, else 400. Camera indices and stream URLs pass
through. Tested (`test_path_traversal_is_blocked`).

**Upload limits** — enforced by size (`MAX_UPLOAD_MB`, default 200) and by extension allow-list.
The crowdsource endpoint accepts images only; a video there is a 415. Empty files are a 400.
nginx enforces `client_max_body_size` before the request reaches the app.

**Query validation** — every filter and enum is validated by Pydantic/FastAPI. `sort_by` is
constrained by regex to a fixed column set, so it cannot be used for injection.

**NoSQL injection** — queries are built from validated, typed values, never from interpolated
strings. User-supplied `search` reaches MongoDB as a `$regex` value; it is length-capped at 120
characters. A production system should also anchor or escape it — a pathological pattern is a ReDoS
risk against the database.

**Error handling** — unhandled exceptions log the traceback server-side and return a terse message.
Stack traces are never sent to the client.

**Credentials in logs** — MongoDB connection strings are redacted before logging.

---

## Known gaps

| Gap | Impact | Mitigation before deployment |
|---|---|---|
| No rate limiting | Crowdsource endpoint can be flooded; each submission runs inference | Per-IP and per-account limits at the gateway; a queue with backpressure |
| No CSRF protection | Token auth in a header is not cookie-based, so the risk is low today; it changes if session cookies are adopted | Standard CSRF tokens if cookies are introduced |
| Media served without authorisation | Anyone with a snapshot URL can read it | Signed, expiring URLs from object storage |
| No content validation beyond extension | A file can be renamed `.jpg` | Verify magic bytes; decode in a sandbox |
| CORS allows configured origins with credentials | Fine locally; must be tightened in production | Explicit origin list, no wildcards |
| No HTTPS in the compose stack | Traffic is plaintext | TLS termination at the load balancer/ingress |
| Secrets via environment variables | Adequate for a prototype | A secrets manager (Vault, AWS Secrets Manager, Azure Key Vault) |
| No dependency scanning in CI | Vulnerable transitive deps could go unnoticed | `pip-audit` / `npm audit` in CI, with pinned versions already in place |
| Ingestion runs in-process | A malicious or malformed video could stress the API process | Isolated worker with resource limits |

---

## Deployment checklist

- [ ] `AUTH_ENABLED=true`, backed by a real identity provider
- [ ] `ALLOW_MEMORY_DB_FALLBACK=false` — never silently lose data in production
- [ ] MongoDB with authentication, TLS, and network isolation (not the open port in the dev compose file)
- [ ] TLS everywhere; HSTS
- [ ] Snapshots and uploads in object storage with signed URLs
- [ ] Rate limiting on all write endpoints, especially crowdsource
- [ ] Face and number-plate blurring on ingestion (see [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md))
- [ ] Retention policy implemented and enforced
- [ ] Centralised logging and alerting; audit log for authorisation decisions
- [ ] Backups with a tested restore
- [ ] Dependency scanning in CI
