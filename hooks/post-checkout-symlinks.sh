#!/bin/bash
set -euo pipefail

# pre-commit sets PRE_COMMIT_CHECKOUT_TYPE (1 = branch/worktree, 0 = file)
[ "${PRE_COMMIT_CHECKOUT_TYPE:-}" = "1" ] || exit 0

# Are we inside a worktree?  In a worktree --git-dir ≠ --common-dir.
git_dir="$(git rev-parse --git-dir)"
common_dir="$(git rev-parse --common-dir)"
[ "$git_dir" = "$common_dir" ] && exit 0          # main repo — nothing to do

main_repo="$(git config --get core.mainRepo 2>/dev/null)" || exit 0
main_repo="${main_repo%/}"                               # strip trailing slash

# Relative path from this worktree back to the main repo root.
# Worktrees live at <repo>/.worktrees/<name>/, so "../../" reaches the root.
# Strip the main-repo prefix; depth = slashes remaining + 1.
wt_root="$(pwd)"
main_rel="${wt_root#"$main_repo"}"
main_rel="${main_rel#/}"
depth=$(echo "$main_rel" | tr -cd '/' | wc -c)
depth=$((depth + 1))
prefix=""
for ((i = 0; i < depth; i++)); do prefix="../${prefix}"; done

for target in sbt.db data; do
  if [ -L "$target" ] || [ -e "$target" ]; then continue; fi
  ln -s "${prefix}${target}" "$target"
done
