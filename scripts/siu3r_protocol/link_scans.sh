# link fully-extracted SIU3R val scans into the SIU3R-layout root (idempotent)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${DATA_ROOT:-$root/data}/siu3r_val_full"
R="${DATA_ROOT:-$root/data}/siu3r_protocol_root/val"
mkdir -p "$R"
for d in "$S"/scene*/.extracted; do
    [[ -f "$d" ]] || continue
    s="$(basename "$(dirname "$d")")"
    [[ -e "$R/$s" ]] || ln -s "$S/$s" "$R/$s"
done
ls "$R" | wc -l
