# Deploying MemAssist

This directory deploys the published GHCR images behind Caddy, with HTTPS and a
password, on one small ARM server. It targets Oracle Cloud's Always Free A1
instance because that is the only free tier that fits — but nothing here is
Oracle-specific: it is a Linux box with Docker and ports 80/443, and any such
box works.

Read [Caveats](#caveats) before you start. There is a real chance you cannot get
the machine at all.

Files:

| File | Purpose |
| --- | --- |
| `docker-compose.deploy.yml` | The 5 services. Pulls images, builds nothing. |
| `Caddyfile` | TLS, basic auth, `/api` → api, everything else → web. |
| `.env.example` | Copy to `.env` on the server and fill in. |

## 1. Create the instance

Oracle Cloud → Compute → Instances → Create.

- **Shape:** Ampere `VM.Standard.A1.Flex`. Set it to **2 OCPU / 12 GB**. That is
  the whole free allowance as of June 2026, when Oracle halved it from 4 OCPU /
  24 GB. Older guides still say 4/24; they are out of date and the console will
  reject it.
- **Image:** Ubuntu 24.04, the **aarch64** build. A1 is ARM — this is why the
  publish workflow builds `linux/arm64` at all.
- **Boot volume:** 50 GB is plenty and within the free 200 GB.
- **SSH key:** paste a public key. Keep the private half; GitHub needs it later.

Note the public IP when it finishes.

## 2. Open the ports — in both places

Two firewalls sit between the internet and the container, and missing either one
produces the same silent hang. Oracle's Ubuntu images ship iptables rules that
drop inbound traffic regardless of what the cloud-side list says.

**Cloud side:** VCN → Security Lists → Default → Add Ingress Rules, source
`0.0.0.0/0`, TCP ports 80 and 443.

**Host side:**

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp          # do this BEFORE enabling, or you lock yourself out
sudo ufw enable
```

## 3. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker                  # or log out and back in
docker compose version         # expect v2.x
```

## 4. A domain name

Caddy needs a name to obtain a certificate for; it cannot issue one for a bare
IP. A free [DuckDNS](https://www.duckdns.org) subdomain is fine — create
`something.duckdns.org`, point it at the instance's public IP, and wait for it
to resolve:

```bash
dig +short something.duckdns.org      # must print your instance IP
```

Do not skip that check. If DNS has not propagated, Caddy's certificate request
fails and the site serves nothing, which reads like a broken deploy.

## 5. First bring-up, by hand

Do this manually once. The GitHub workflow only ever repeats what you prove
works here.

```bash
sudo mkdir -p /opt/memassist && sudo chown "$USER" /opt/memassist
cd /opt/memassist

# From your laptop, in a checkout of this repo:
#   scp deploy/docker-compose.deploy.yml deploy/Caddyfile ubuntu@<ip>:/opt/memassist/
#   scp deploy/.env.example ubuntu@<ip>:/opt/memassist/.env

# Generate the basic-auth hash:
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'pick-a-password'

nano .env      # DEPLOY_DOMAIN, CADDY_USER, CADDY_HASH, POSTGRES_PASSWORD,
               # and at least one provider key

docker compose -f docker-compose.deploy.yml pull
docker compose -f docker-compose.deploy.yml up -d
docker compose -f docker-compose.deploy.yml ps    # expect 5 services, healthy
```

First pull takes a few minutes: the api image carries CPU torch and a baked
copy of bge-small.

Then, from anywhere:

```bash
curl -u user:password https://your-domain/api/healthz
# {"status":"ok","backend":"postgres"}
```

Open `https://your-domain` in a browser. It should prompt for the password and
then show the UI.

If the certificate never arrives, `docker compose -f docker-compose.deploy.yml
logs caddy` says why — nearly always DNS, or port 80 closed in one of the two
firewalls.

## 6. Hand it to GitHub

Repository → Settings → Environments → **New environment**, named exactly
`demo`. The name is load-bearing: `.github/workflows/deploy.yml` reads its
secrets from an environment with that name, and skips cleanly when they are
absent.

Add these environment secrets:

| Secret | Value |
| --- | --- |
| `DEPLOY_HOST` | the instance's public IP or your domain |
| `DEPLOY_USER` | `ubuntu` |
| `DEPLOY_SSH_KEY` | the **private** key, whole file including the BEGIN/END lines |
| `DEPLOY_DOMAIN` | *(optional)* the domain the health check probes |
| `DEPLOY_BASIC_AUTH` | *(optional)* `user:password`, so the check can get past basic auth |

The last two are optional but the health check is much weaker without them: it
falls back to `DEPLOY_HOST`, and basic auth will answer 401, which the check
correctly treats as a failure.

From then on, publishing a GitHub release deploys that tag, and
Actions → deploy → Run workflow deploys any tag you name.

## Caveats

Be honest with yourself about this tier before building on it. **Capacity:**
A1 is chronically out of stock in popular regions, and "Out of host capacity"
on creation is the normal experience, not a bug — people retry for days, or
script it. **Idle reclaim:** Oracle reclaims Always Free compute it considers
idle (roughly: under ~20% CPU, ~20% network and ~10% memory across a 7-day
window). A demo nobody visits is exactly that shape, so add a keep-alive:

```bash
# crontab -e  — a cheap heartbeat that also proves the stack still answers
*/10 * * * * curl -fsS -u user:password https://your-domain/api/healthz >/dev/null 2>&1
```

**Payment card:** signup requires a card and an identity check even though the
Always Free resources never charge it. **Not a guarantee:** Oracle can change
the free shape again; it already did in June 2026.

If you cannot get an A1 instance, nothing is lost and nothing here is wasted.
**The supported deployment remains local `docker compose`**, exactly as the root
README describes, and this workflow targets any Linux box you later own — a
Hetzner or DigitalOcean VM, a home server, a Raspberry Pi 5 (the arm64 images
are already built for it). Only steps 1 and 2 are Oracle-specific; 3 through 6
are the same anywhere.

## Rolling back

```bash
cd /opt/memassist
sed -i 's|^MEMASSIST_VERSION=.*|MEMASSIST_VERSION=1.0.0|' .env
docker compose -f docker-compose.deploy.yml up -d
```

Images are immutable per tag, so a previous tag is a previous deployment. The
database is not: a release that migrates schema is not undone by this.
