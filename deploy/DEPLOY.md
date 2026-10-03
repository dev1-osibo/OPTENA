# Deployment runbook — reviewer instance

## Target, as measured

| | |
|---|---|
| Instance | `i-062f61877c1eaa4d7`, `c7i-flex.large`, `us-east-1b` |
| Public IP | `98.87.133.69` |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-1004-aws |
| Python | CPython 3.14.4 (system; no 3.11/3.12/3.13 present) |
| CPU / RAM | 2 vCPU, 3.7 GiB, **no swap** |
| Disk | 58 GB, 45 GB used, **13 GB free (78%)** |
| Firewall | `ufw` **inactive** — access is governed solely by security group `launch-wizard-2` |
| Uptime at survey | 146 days, load average 0.00 |
| sudo | passwordless for `ubuntu` |

## What is already on the box — do not disturb

This is the **Cortafy** host, not the old research box (`98.90.192.78`) referenced in
`PROJECT_STATUS_AND_SALVAGE_ASSESSMENT.md`. No MIMIC/eICU/NHANES medical data was found
here. What is running:

| Service | Detail |
|---|---|
| `cortafy.service` | uvicorn `main:app` on `0.0.0.0:8000`, from `/home/ubuntu/cortafy/venv` |
| `postgresql` | `127.0.0.1:5432` |
| `nginx` | ports 80 and 443, three live vhosts |

The three vhosts are `cortafy` (**default_server** on 80/443, plus `cortafy.com`),
`payfonte.cortafy.com`, and `pentrestglobal.com`. Treat all three as production.

Also present, unrelated to this deploy: `~/tft_training` (1.5 GB, the TFT/DCPLOS work
pushed by the old `deploy_to_aws.bat`) and `~/failure_data` (**37 GB** of Backblaze data,
the reason the disk is 78% full).

**The Optena app is not on this box.** There is no `~/optena`, no `optena1.py` and no
`dcplos_engine.py` anywhere under `/home/ubuntu`.

## Why this design

Streamlit ships **no authentication**. Exposing it directly on a public port would put
an unauthenticated app on a host that also serves three production sites. So:

- Streamlit binds **`127.0.0.1:8501`** only — never `0.0.0.0`.
- A **new** nginx site listens on **8080** and applies HTTP basic auth, proxying to
  loopback with the websocket upgrade headers Streamlit requires.
- Port 8080 is a separate listener in a **new** site file. No existing vhost is edited
  and ports 80/443 behave exactly as before.
- `systemd` caps the service at `MemoryMax=1200M` and `CPUQuota=120%` so a runaway
  session cannot starve cortafy on a 2 vCPU / 3.7 GiB box with no swap.
- `nginx -t` runs before any reload; if it fails the script aborts without reloading,
  leaving production untouched.

## Feasibility, verified not assumed

A throwaway venv was built in `/tmp` on this instance and then deleted:

```
venv creation                 4s
pip install of the full set  17s
venv footprint              644 MB
all 7 packages import OK    on CPython 3.14.4
```

Resolved versions: streamlit 1.65.0, pandas 3.0.6, numpy 2.5.3, matplotlib 3.11.2,
plotly 7.1.0, prophet 1.4.0, kaleido 1.4.0.

**The original `requirements.txt` cannot be used here.** Its pins (streamlit 1.37.1,
numpy 1.26.4, pandas 2.2.2, prophet 1.1.6, pmdarima 2.0.4) have no CPython 3.14 wheels.
The lock in this repo is the version-raised, 3.14-verified replacement.

## Steps

### 1. Copy the app up

From the workstation:

```powershell
$key = "C:\Users\babas\Downloads\cortafy-key.pem"
$src = "c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\optena-app"
scp -i $key -r $src ubuntu@98.87.133.69:/home/ubuntu/optena-app
```

### 2. Run the deploy script

```bash
ssh -i cortafy-key.pem ubuntu@98.87.133.69
cd /home/ubuntu/optena-app
OPTENA_USER=reviewer OPTENA_PASS='<pick-a-strong-password>' bash deploy/deploy_optena.sh
```

The script is idempotent. On re-runs it keeps the existing credentials file; delete
`/etc/nginx/.optena_htpasswd` to rotate.

### 3. Open the port — manual, and deliberately not automated

`aws` CLI is **not installed** on the instance, so the security group cannot be changed
from there. In the AWS console:

> EC2 → Instances → `i-062f61877c1eaa4d7` → Security → security group
> `launch-wizard-2` → Inbound rules → Edit → Add rule
> **Custom TCP, port 8080, source = the reviewer's IP /32**

Use the reviewer's specific IP rather than `0.0.0.0/0`. Basic auth over plain HTTP sends
the password base64-encoded, not encrypted, so source restriction is doing real work here.

### 4. Verify from outside

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://98.87.133.69:8080/            # expect 401
curl -s -o /dev/null -w '%{http_code}\n' -u reviewer:<pass> http://98.87.133.69:8080/  # expect 200
```

Then hand the reviewer `http://98.87.133.69:8080/`, the credentials, and
`REVIEWER_GUIDE.md`.

## Residual risks, stated plainly

| Risk | Severity | Mitigation in place |
|---|---|---|
| Shared host with three production vhosts | high if mishandled | separate port, new site file, `nginx -t` gate, no edits to existing configs, cortafy never restarted |
| 13 GB free disk | medium | venv is 644 MB; script aborts below 2 GB free |
| 3.7 GiB RAM, no swap | medium | `MemoryMax=1200M`, `CPUQuota=120%` |
| Basic auth over HTTP, not HTTPS | medium | restrict source IP; the app holds no real data. For HTTPS you would need a DNS name and certificate, since the existing cert is for `cortafy.com` and would mismatch an IP |
| Streamlit has no app-level auth | medium | all access control is at nginx |
| App written for streamlit 1.37 / plotly 5.15, running on 1.65 / 7.1 | **unverified** | dependency imports are confirmed; the app's own runtime behaviour on these newer majors has **not** been exercised — do a smoke test before handing over the URL |

## Rollback

```bash
sudo systemctl disable --now optena
sudo rm -f /etc/systemd/system/optena.service
sudo rm -f /etc/nginx/sites-enabled/optena /etc/nginx/sites-available/optena
sudo rm -f /etc/nginx/.optena_htpasswd
sudo systemctl daemon-reload
sudo nginx -t && sudo systemctl reload nginx
rm -rf /home/ubuntu/optena-app
```

Then remove the port 8080 inbound rule from `launch-wizard-2`. This returns the host to
exactly its pre-deploy state; nothing in the procedure modifies cortafy, postgresql, or
the three existing vhosts.
