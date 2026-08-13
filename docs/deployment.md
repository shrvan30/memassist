# Deployment

Local `docker compose up` is the supported deployment. Everything beyond
it is built, gated, and documented — honestly, including the free-tier
reality.

## Local (supported)

```bash
cp .env.example .env   # at least one provider key
docker compose up --build
```

Four services come up health-gated (postgres -> memory-mcp -> api ->
web). **Postgres publishes on host `15432`, deliberately not 5432**: a
native Postgres or another compose stack with `restart: always` commonly
owns 5432 and reclaims it at every engine boot — colliding there kills
`compose up` before anything runs. Override with `POSTGRES_HOST_PORT`;
in-network services always use `postgres:5432` and do not care.

## Images

A `v*` release tag runs the full pipeline and, on green, publishes
**multi-arch (linux/amd64 + linux/arm64)** images for `memassist-api` and
`memassist-web` to GHCR — arm64 exists because the one genuinely free
deployment target (Oracle Always Free A1) is ARM. The api image installs
CPU-only torch and bakes the embedding model into a layer, so containers
never pay the 135MB download at start. The publish job is tag-gated and
skips on ordinary pushes.

## The gated deploy pipeline

`deploy/docker-compose.deploy.yml` runs the same four services from
pulled GHCR images plus **Caddy** in front: automatic HTTPS, HTTP basic
auth, `/api` -> api, everything else -> web; only 80/443 published. The
published web image is built with `NEXT_PUBLIC_API_BASE=/api` (relative,
same-origin) — a localhost value baked into a public image would send
every visitor's browser to their own machine.

`.github/workflows/deploy.yml` fires on release-publish or manual
dispatch, gated by a GitHub environment named `demo`: with no secrets it
skips grey (never red). With `DEPLOY_HOST` / `DEPLOY_USER` /
`DEPLOY_SSH_KEY` set, it SSHes to the box, pulls, brings the stack up,
and fails loudly if the health endpoint does not answer through Caddy.

## The free-tier reality (checked, dated, honest)

As of mid-2026: Render's free tier (512MB / 0.1 CPU) cannot hold torch;
Fly and Koyeb closed free tiers to new accounts; HF bills Docker Spaces.
The one real option is **Oracle Always Free A1** (halved June 2026 to 2
OCPU / 12GB — still ample): card-at-signup, a regional capacity lottery,
and an idle-reclaim policy (keep a light cron alive). The verified
step-by-step, including OCI firewall rules and DuckDNS, is in
`deploy/README.md`. If the lottery says no: the supported deployment
remains local compose, and this pipeline targets any Linux box you later
own.

## Do not deploy publicly without auth

A public instance without the Caddy basic-auth layer means strangers
spending your provider quotas and writing into your memory. The compose
deploy file refuses to start without `POSTGRES_PASSWORD` and the Caddy
credentials for exactly this reason.
