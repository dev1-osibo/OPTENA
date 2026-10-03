#!/usr/bin/env bash
#
# Deploy the Optena demo app to the reviewer instance.
#
# Run ON the instance, from the directory containing this script's parent repo:
#   bash deploy/deploy_optena.sh
#
# Idempotent: safe to re-run. It never touches the cortafy, payfonte.cortafy.com or
# pentrestglobal.com nginx configs, never restarts cortafy/postgresql, and never
# modifies ports 80 or 443.
#
set -euo pipefail

APP_DIR=/home/ubuntu/optena-app
VENV="$APP_DIR/venv"
LOG_DIR=/var/log/optena
PORT_INTERNAL=8501
PORT_PUBLIC=8080

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "Preflight"
[ -f "$APP_DIR/optena1.py" ]      || die "optena1.py not found in $APP_DIR"
[ -f "$APP_DIR/dcplos_engine.py" ] || die "dcplos_engine.py not found (Tab 5 needs it)"
[ -f "$APP_DIR/requirements.txt" ] || die "requirements.txt not found in $APP_DIR"
[ -f "$APP_DIR/optena.jpg" ]       || die "optena.jpg not found (app loads it by relative path)"

echo "python3: $(python3 --version)"
AVAIL_KB=$(df --output=avail / | tail -1)
echo "free disk: $(( AVAIL_KB / 1024 / 1024 )) GB"
[ "$AVAIL_KB" -gt 2097152 ] || die "need at least 2 GB free for the venv (644 MB) plus headroom"

# Confirm we are not about to collide with anything already listening.
if ss -tln 2>/dev/null | grep -q ":$PORT_INTERNAL "; then
  echo "NOTE: something already listens on $PORT_INTERNAL — assuming it is a previous optena deploy"
fi
if ss -tln 2>/dev/null | grep -q ":$PORT_PUBLIC "; then
  echo "NOTE: something already listens on $PORT_PUBLIC"
fi

say "Production services before deploy (must be unchanged at the end)"
systemctl is-active cortafy nginx postgresql || true

# ---------------------------------------------------------------- virtualenv
say "Building virtualenv at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip --quiet
"$VENV/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
echo "installed:"
"$VENV/bin/pip" list 2>/dev/null | grep -Ei "^(streamlit|pandas|numpy|matplotlib|plotly|prophet|kaleido) " || true

say "Import smoke test"
"$VENV/bin/python" - <<'PY'
import importlib, sys
bad = []
for m in ("streamlit", "pandas", "numpy", "matplotlib", "plotly", "prophet", "kaleido"):
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
if bad:
    print("IMPORT FAILURES:"); [print("  " + b) for b in bad]; sys.exit(1)
print("all dependencies import cleanly on", sys.version.split()[0])
PY

say "Compile check on the app itself"
"$VENV/bin/python" -m py_compile "$APP_DIR/optena1.py" "$APP_DIR/dcplos_engine.py"
echo "optena1.py and dcplos_engine.py compile"

# ---------------------------------------------------------------- logging
say "Log directory"
sudo mkdir -p "$LOG_DIR"
sudo chown ubuntu:ubuntu "$LOG_DIR"

# ---------------------------------------------------------------- systemd
say "Installing systemd unit (binds 127.0.0.1 only)"
sudo cp "$APP_DIR/deploy/optena.service" /etc/systemd/system/optena.service
sudo systemctl daemon-reload
sudo systemctl enable optena.service
sudo systemctl restart optena.service
sleep 6
systemctl is-active optena.service || {
  echo "--- journal ---"; sudo journalctl -u optena.service -n 40 --no-pager
  die "optena.service did not start"
}

say "Local reachability check (before any public exposure)"
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS -o /dev/null "http://127.0.0.1:$PORT_INTERNAL/_stcore/health" 2>/dev/null; then
    echo "streamlit health endpoint OK on 127.0.0.1:$PORT_INTERNAL"; break
  fi
  [ "$i" = 10 ] && die "streamlit did not become healthy on 127.0.0.1:$PORT_INTERNAL"
  sleep 3
done

# ---------------------------------------------------------------- nginx + auth
say "nginx front end on port $PORT_PUBLIC with basic auth"
if [ ! -f /etc/nginx/.optena_htpasswd ]; then
  if [ -z "${OPTENA_USER:-}" ] || [ -z "${OPTENA_PASS:-}" ]; then
    die "set OPTENA_USER and OPTENA_PASS before first run, e.g.
     OPTENA_USER=reviewer OPTENA_PASS='<strong-password>' bash deploy/deploy_optena.sh"
  fi
  command -v htpasswd >/dev/null || sudo apt-get install -y apache2-utils
  sudo htpasswd -bc /etc/nginx/.optena_htpasswd "$OPTENA_USER" "$OPTENA_PASS"
  sudo chown root:www-data /etc/nginx/.optena_htpasswd
  sudo chmod 640 /etc/nginx/.optena_htpasswd
  echo "created credentials for user '$OPTENA_USER'"
else
  echo "existing /etc/nginx/.optena_htpasswd kept (delete it to rotate credentials)"
fi

sudo cp "$APP_DIR/deploy/nginx-optena.conf" /etc/nginx/sites-available/optena
sudo ln -sfn /etc/nginx/sites-available/optena /etc/nginx/sites-enabled/optena

say "Validating nginx config (aborts without reloading if invalid)"
sudo nginx -t || die "nginx config test failed — nothing was reloaded, production untouched"
sudo systemctl reload nginx

say "Authenticated fetch through nginx"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_PUBLIC/" || true)
echo "unauthenticated request -> HTTP $code (expect 401)"
[ "$code" = "401" ] || echo "WARNING: expected 401, got $code — check auth_basic"

say "Production services after deploy (compare with the list above)"
systemctl is-active cortafy nginx postgresql || true

cat <<EOF

==================================================================
Deploy complete.

  app dir        : $APP_DIR
  venv           : $VENV
  service        : optena.service  (bound to 127.0.0.1:$PORT_INTERNAL)
  public port    : $PORT_PUBLIC via nginx, HTTP basic auth
  logs           : $LOG_DIR/app.log, $LOG_DIR/error.log
                   /var/log/nginx/optena_{access,error}.log

REMAINING MANUAL STEP — the app is NOT yet reachable from the internet.
Open TCP $PORT_PUBLIC in security group launch-wizard-2 (instance i-062f61877c1eaa4d7,
us-east-1b). Restrict the source to the reviewer's IP rather than 0.0.0.0/0.

Then the reviewer uses:  http://98.87.133.69:$PORT_PUBLIC/

Useful commands:
  sudo systemctl status optena
  sudo journalctl -u optena -f
  sudo systemctl restart optena
  tail -f $LOG_DIR/app.log

To remove completely:
  sudo systemctl disable --now optena
  sudo rm /etc/systemd/system/optena.service /etc/nginx/sites-enabled/optena
  sudo rm /etc/nginx/sites-available/optena /etc/nginx/.optena_htpasswd
  sudo systemctl daemon-reload && sudo nginx -t && sudo systemctl reload nginx
  rm -rf $APP_DIR
==================================================================
EOF
