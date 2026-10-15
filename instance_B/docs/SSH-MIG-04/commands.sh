# Executed from Uni3R root with system Windows OpenSSH to 010-new.
source instance_B/docs/SSH-MIG-03/isolation.sh
export TMPDIR="$B_ROOT/docs/SSH-MIG-04/tmp" TMP="$B_ROOT/docs/SSH-MIG-04/tmp" TEMP="$B_ROOT/docs/SSH-MIG-04/tmp" XDG_CACHE_HOME="$B_ROOT/docs/SSH-MIG-04/cache"
"$B_ROOT/.venv/bin/python" -B "$B_ROOT/docs/SSH-MIG-04/run_mig04.py"
# Do not rerun: exclusive outputs and the exhausted 8/8 persistent ledger must be preserved.
