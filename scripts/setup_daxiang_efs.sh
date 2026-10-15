#!/usr/bin/env bash
# Run after cloning. Only links absent/empty directories; never replaces data.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
storage="${UNI3R_EFS_ROOT:-/file_system/efs/checkpoint/intern/ying.wu}"
if [[ "$(findmnt -n -o TARGET -T "$storage")" != /file_system/efs ]]; then
    echo "Expected an EFS directory mounted at /file_system/efs: $storage" >&2
    exit 1
fi
mkdir -p "$storage"/{checkpoints,data,cache,tmp,outputs,logs,test_results,envs}
for item in checkpoints data .cache outputs logs test_results; do
    target="$storage/$item"
    [[ "$item" == .cache ]] && target="$storage/cache"
    link="$root/$item"
    if [[ -L "$link" ]] && [[ "$(readlink -f "$link")" == "$target" ]]; then
        continue
    fi
    if [[ -d "$link" && ! -L "$link" ]]; then
        rmdir "$link" # Fails rather than deleting existing contents.
    elif [[ -e "$link" || -L "$link" ]]; then
        echo "Refusing to replace $link" >&2
        exit 1
    fi
    ln -s "$target" "$link"
done
patch="$root/patches/lang-seg-reproduction.patch"
if git -C "$root/submodules/lang_seg" apply --reverse --check "$patch" 2>/dev/null; then
    echo 'LSeg reproduction patch already applied'
else
    git -C "$root/submodules/lang_seg" apply --check "$patch"
    git -C "$root/submodules/lang_seg" apply "$patch"
fi
echo "EFS storage ready: $storage"
echo 'Run: source scripts/env_daxiang.sh'
