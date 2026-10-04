#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-$REPO_ROOT}"
PATCH_ROOT="${PATCH_ROOT:-$REPO_ROOT}"
# When set (by setup_worktree.sh), mutable dependency checkouts live under the
# worktree. Existing dependency symlinks are replaced with private clones,
# seeded from DEPENDENCY_ROOT when possible so setup does not mutate the shared
# checkout or require a network fetch for already-present objects.
WORKTREE_ROOT="${WORKTREE_ROOT:-}"


repo_path_for() {
    local name="$1"
    if [ -n "$WORKTREE_ROOT" ]; then
        echo "$WORKTREE_ROOT/$name"
    else
        echo "$DEPENDENCY_ROOT/$name"
    fi
}

prepare_worktree_dependency() {
    local name="$1"
    local repo_url="$2"
    [ -n "$WORKTREE_ROOT" ] || return 0
    [ "$WORKTREE_ROOT" != "$DEPENDENCY_ROOT" ] || return 0

    local worktree_path="$WORKTREE_ROOT/$name"
    local shared_path="$DEPENDENCY_ROOT/$name"

    if [ -L "$worktree_path" ]; then
        echo "Replacing shared dependency symlink with private checkout: $name"
        unlink "$worktree_path"
    fi

    if [ -e "$worktree_path" ] && [ ! -e "$worktree_path/.git" ]; then
        echo "Error: $worktree_path exists but is not a git checkout." >&2
        exit 1
    fi
    if [ -e "$worktree_path/.git" ]; then
        return 0
    fi

    if [ -e "$shared_path/.git" ]; then
        echo "Seeding worktree-local $name from $shared_path"
        mkdir -p "$(dirname "$worktree_path")"
        git clone -q --no-hardlinks "$shared_path" "$worktree_path"
        git -C "$worktree_path" remote set-url origin "$repo_url"
    fi
}

hash_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1
    else
        sha256sum "$1" 2>/dev/null | cut -d' ' -f1
    fi
}

deps=(
    "bergson|https://github.com/EleutherAI/bergson.git|a4f7f246a5dfeae01e650346c6d00cac51c37c20|patches/bergson-tokenizer-revision.patch"
    "olmes|https://github.com/allenai/olmes.git|b532dd59a71004d88f8152788d79cec617c8eff6|patches/olmes-fix.patch"
)

ensure_clean_for_checkout() {
    local repo_path="$1"
    local name="$2"
    local dirty
    dirty="$(git -C "$repo_path" status --porcelain | grep -vE '^\?\? \.DS_Store$' || true)"
    if [ -n "$dirty" ]; then
        echo "Error: $name has local changes; cannot switch commits automatically." >&2
        echo "$dirty" >&2
        exit 1
    fi
}

patch_targets() {
    local patch_path="$1"
    sed -n 's/^diff --git a\/\(.*\) b\/.*/\1/p' "$patch_path"
}

patch_file_state() {
    local repo_path="$1"
    local patch_path="$2"
    local target="$3"
    if git -C "$repo_path" apply --reverse --check --include="$target" "$patch_path" >/dev/null 2>&1; then
        echo "applied"
        return
    fi
    if git -C "$repo_path" apply --check --include="$target" "$patch_path" >/dev/null 2>&1; then
        echo "missing"
        return
    fi
    # The checks above match three lines of surrounding context, so an edit
    # anywhere near a hunk makes an applied patch look indistinguishable from a
    # corrupted one.  The vendored trees are shared across every worktree and do
    # accumulate such edits: one added import block in olmes shifted the context
    # around an already-applied hunk and failed bootstrap in every worktree at
    # once, with the patch itself perfectly intact.  Retry with one line of
    # context so drift near a hunk no longer masquerades as divergence.  Reverse
    # still runs first, and it only succeeds when the patch's own added lines are
    # present, so this loosens the context match without loosening the content
    # match.
    if git -C "$repo_path" apply --reverse --check -C1 --include="$target" "$patch_path" >/dev/null 2>&1; then
        echo "applied"
        return
    fi
    if git -C "$repo_path" apply --check -C1 --include="$target" "$patch_path" >/dev/null 2>&1; then
        echo "missing"
        return
    fi
    echo "unknown"
}

apply_patch_if_needed() {
    local repo_path="$1"
    local patch_path="$2"
    local name="$3"
    local target
    local state
    local missing_targets=()
    local unknown_targets=()
    local apply_args=()

    while IFS= read -r target; do
        [ -n "$target" ] || continue
        state="$(patch_file_state "$repo_path" "$patch_path" "$target")"
        case "$state" in
            applied) ;;
            missing) missing_targets+=("$target") ;;
            *) unknown_targets+=("$target") ;;
        esac
    done < <(patch_targets "$patch_path")

    if [ ${#unknown_targets[@]} -gt 0 ]; then
        echo "Error: $name has diverged from tracked patch $(basename "$patch_path")" >&2
        printf 'Unclassified patch targets:\n' >&2
        printf '  %s\n' "${unknown_targets[@]}" >&2
        echo "Refresh the tracked patch or restore the vendored repo to the pinned patched state." >&2
        exit 1
    fi

    if [ ${#missing_targets[@]} -eq 0 ]; then
        echo "Patch already applied for $name: $(basename "$patch_path")"
        return
    fi

    echo "Applying patch for $name: $(basename "$patch_path")"
    for target in "${missing_targets[@]}"; do
        apply_args+=(--include="$target")
    done
    git -C "$repo_path" apply "${apply_args[@]}" "$patch_path"
}

patches_all_marked() {
    local repo_path="$1"
    local patch_list="$2"
    [ -n "$patch_list" ] || return 0
    local p_arr
    IFS=',' read -r -a p_arr <<<"$patch_list"
    for patch_rel in "${p_arr[@]}"; do
        [ -n "$patch_rel" ] || continue
        local patch_path="$PATCH_ROOT/$patch_rel"
        local marker_file="$repo_path/.patched_$(basename "$patch_rel" .patch)"
        local patch_hash
        patch_hash="$(hash_file "$patch_path")"
        if [ ! -f "$marker_file" ] || [ "$(cat "$marker_file")" != "$patch_hash" ]; then
            return 1
        fi
    done
    return 0
}

# Everything above is a function definition; everything below clones, checks out
# and patches real trees.  Sourcing this file to test one of those functions
# should not do any of that, so honour a library-only mode and stop here.
if [ -n "${BOOTSTRAP_LIBRARY_ONLY:-}" ]; then
    return 0
fi

for spec in "${deps[@]}"; do
    IFS="|" read -r name repo_url commit patch_list <<<"$spec"
    prepare_worktree_dependency "$name" "$repo_url"
    repo_path="$(repo_path_for "$name")"

    echo "==> $name"
    if [ "$repo_path" = "$WORKTREE_ROOT/$name" ]; then
        echo "  (using worktree-local $name)"
    fi

    if [ -d "$repo_path/.git" ]; then
        current_commit="$(git -C "$repo_path" rev-parse HEAD 2>/dev/null || true)"
        if [ "$current_commit" = "$commit" ] && patches_all_marked "$repo_path" "$patch_list"; then
            echo "$name already at pinned commit with patches applied"
            continue
        fi
    fi

    if [ ! -d "$repo_path/.git" ]; then
        echo "Cloning $repo_url into $repo_path"
        git clone "$repo_url" "$repo_path"
    fi

    if ! git -C "$repo_path" cat-file -e "$commit^{commit}" >/dev/null 2>&1; then
        echo "Fetching $name to resolve pinned commit"
        git -C "$repo_path" fetch --tags origin
    fi

    current_commit="$(git -C "$repo_path" rev-parse HEAD)"
    if [ "$current_commit" != "$commit" ]; then
        ensure_clean_for_checkout "$repo_path" "$name"
        echo "Checking out $name at $commit"
        git -C "$repo_path" checkout "$commit"
    else
        echo "$name already at pinned commit"
    fi

    if [ -n "$patch_list" ]; then
        IFS=',' read -r -a patches <<<"$patch_list"
        for patch_rel in "${patches[@]}"; do
            [ -n "$patch_rel" ] || continue
            patch_path="$PATCH_ROOT/$patch_rel"
            apply_patch_if_needed "$repo_path" "$patch_path" "$name"
            echo "$(hash_file "$patch_path")" > "$repo_path/.patched_$(basename "$patch_rel" .patch)"
        done
    fi
done

echo "Local dependencies are ready in $DEPENDENCY_ROOT"
