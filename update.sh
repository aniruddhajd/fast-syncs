#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# fast-syncs update script — pulls the latest version from
# GitHub and re-installs Python dependencies.
#
# Run this whenever you want to grab the newest features.
#
# The whole body lives in main(): bash reads scripts from disk as it
# executes them, so `git pull` rewriting THIS file mid-run could
# otherwise derail the remaining lines. A fully-parsed function (plus
# `exit` on the same line as the call) makes the run immune to that.
# ──────────────────────────────────────────────────────────────
set -e

# The newest RELEASE, not the tip of main. This URL is stable: GitHub always
# redirects it to the fast-syncs.zip asset of the latest published release
# (see .github/workflows/release.yml), so users move between versions that
# were deliberately shipped rather than whatever was committed last.
#
# To install a specific older version instead, download its fast-syncs.zip
# from https://github.com/aniruddhajd/fast-syncs/releases and unzip it
# over this folder — settings and venv are preserved either way.
ZIP_URL="https://github.com/aniruddhajd/fast-syncs/releases/latest/download/fast-syncs.zip"

# Set FAST_SYNCS_ZIP_URL to override (e.g. to pin an older release):
#   FAST_SYNCS_ZIP_URL=https://github.com/.../download/v0.12.0/fast-syncs.zip bash update.sh
ZIP_URL="${FAST_SYNCS_ZIP_URL:-$ZIP_URL}"

# Set by `--version X.Y.Z` (the panel's "Roll back…" button): install that one
# release instead of the newest. Empty = a normal update. See parse_args.
PIN_VERSION=""

# Fallback for the window BEFORE the first release exists: the releases/latest
# URL 404s until something has actually been published, and an updater that
# stops working the moment this lands would be worse than the problem it fixes.
# Tried only if the release download fails, so a normal update never touches it.
FALLBACK_ZIP_URL="https://codeload.github.com/aniruddhajd/fast-syncs/zip/refs/heads/production"

# Set when the new files could NOT be fetched, so the final message never
# claims an update that did not happen.
NO_DL=0

# ──────────────────────────────────────────────────────────────
# The file manifest
#
# Every release ships a .fast-syncs-manifest in its root: one repo-relative
# path per line, listing exactly what that release contains (written by
# .github/workflows/release.yml).
#
# The ZIP overlay below only ADDS and OVERWRITES — it can never remove. So a
# file deleted or renamed upstream would otherwise stay on this machine
# forever, and an installed copy slowly drifts into a state that exists on
# nobody else's disk. Worse, a stale .py module keeps shadowing the new code,
# producing bugs that only ever appear for upgraded users.
#
# The fix: keep the manifest around, and on the next update delete whatever
# the PREVIOUS manifest listed that the NEW one no longer does.
MANIFEST=".fast-syncs-manifest"

# ── Delete-safety ─────────────────────────────────────────────
# This is the only code here that removes a user's files, so every check below
# fails CLOSED: anything unexpected means skip that path and carry on. A stale
# file is a far smaller problem than a deleted one.

# Paths that are never deleted, whatever a manifest claims. Most of these are
# already safe by never appearing in a manifest at all (they are gitignored, so
# they are not in the release) — this is the second layer, for the day a
# manifest is wrong.
manifest_protected() {
    case "$1" in
        venv|venv/*|*/venv|*/venv/*)                               return 0 ;;
        sync_pipeline_settings.json|*/sync_pipeline_settings.json) return 0 ;;
        sync_pipeline_settings.json.bak|*/sync_pipeline_settings.json.bak) return 0 ;;
        dubbing/config|dubbing/config/*)                           return 0 ;;
        dubbing/engine/status|dubbing/engine/status/*)             return 0 ;;
        dubbing/data|dubbing/data/*)                               return 0 ;;
        vertex_key.json|*/vertex_key.json)                         return 0 ;;
        service_account.json|*/service_account.json)               return 0 ;;
        .env|.env.*|*/.env|*/.env.*)                               return 0 ;;
        *.pem|*.key)                                               return 0 ;;
        .direct-mode)                                              return 0 ;;
        github_token.txt|*/github_token.txt)                       return 0 ;;
        # The manifest itself: the new one has just been written and is what
        # the NEXT update will diff against.
        "$MANIFEST"|*/"$MANIFEST")                                 return 0 ;;
        # The user's own audio and REAPER projects.
        *.RPP|*.RPP-bak|*.rpp|*.rpp-bak)                           return 0 ;;
        *.wav|*.mp3|*.aif|*.aiff|*.m4a|*.flac|*.ogg|*.opus)        return 0 ;;
    esac
    return 1
}

# A manifest line must be a plain relative path INSIDE this folder. Anything
# else is refused outright rather than normalised: a corrupted — or hostile —
# manifest must not be able to reach a single byte outside the install.
manifest_path_ok() {
    case "$1" in
        ''|.|..)                 return 1 ;;
        /*|~*)                   return 1 ;;  # absolute, or home-relative
        [A-Za-z]:*)              return 1 ;;  # Windows absolute (C:\...)
        *\\*)                    return 1 ;;  # backslash is not a separator here
        ../*|*/../*|*/..)        return 1 ;;  # climbs out of the folder
        ./*|*/./*)               return 1 ;;  # not canonical — refuse, don't fix
        */)                      return 1 ;;  # a directory, not a file
        *)                       return 0 ;;
    esac
}

# A path can look perfectly local and still land elsewhere if one of its parent
# directories is a symlink (say "dubbing" -> /etc, then "dubbing/passwd").
# Walk the parents and refuse if any of them is a link.
manifest_parents_ok() {
    local d
    d=$(dirname -- "$1")
    while [ "$d" != "." ] && [ "$d" != "/" ] && [ -n "$d" ]; do
        if [ -L "$d" ]; then return 1; fi
        d=$(dirname -- "$d")
    done
    return 0
}

# Delete what the previous release shipped and this one no longer does.
#   $1  the manifest this release brought (already in place)
#   $2  the previous install's manifest, saved before the overlay
#   $3  the freshly unpacked release tree — ground truth for "still shipped"
manifest_prune() {
    local new="$1" prev="$2" src="$3"
    local line d removed=0 skipped=0 dirs=""

    # No previous manifest: either a fresh install, or the first update after
    # this feature landed. There is no record of what the older release
    # contained, so DELETE NOTHING — never guess. The new manifest is now on
    # disk, so every update from here on has one.
    if [ ! -f "$prev" ]; then
        echo "[update] First update carrying a file list — nothing to clean up yet."
        return 0
    fi
    # An empty or unreadable new manifest would make every previously shipped
    # path look obsolete at once. Refuse to prune on one.
    if [ ! -s "$new" ]; then
        echo "[update] The new version brought no usable file list — skipping cleanup."
        return 0
    fi

    # Compare against a copy with any CR stripped. If one manifest were written
    # CRLF and the other LF, every single line would look changed — and that
    # asymmetry is exactly how a cleanup turns into a wipe.
    local newlf="$prev.new-lf"
    tr -d '\r' < "$new" > "$newlf" 2>/dev/null || return 0
    [ -s "$newlf" ] || return 0

    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%$'\r'}
        [ -n "$line" ] || continue

        # Still shipped? Then it is not obsolete. Checked against BOTH the new
        # manifest and the unpacked tree: if the manifest were truncated or
        # garbled, the tree still vetoes deleting a file just installed.
        if grep -Fxq -- "$line" "$newlf"; then continue; fi

        if ! manifest_path_ok "$line"; then
            echo "[update]   refusing suspicious manifest path: $line"
            skipped=$((skipped + 1))
            continue
        fi
        if [ -e "$src/$line" ]; then continue; fi
        if manifest_protected "$line"; then
            skipped=$((skipped + 1))
            continue
        fi
        if ! manifest_parents_ok "$line"; then
            echo "[update]   refusing path below a symlinked folder: $line"
            skipped=$((skipped + 1))
            continue
        fi
        # Already gone, or never installed. Directories are never removed
        # here — only the empty-directory sweep below touches those.
        if [ ! -f "$line" ] && [ ! -L "$line" ]; then continue; fi

        if rm -f -- "$line" 2>/dev/null; then
            removed=$((removed + 1))
            dirs="$dirs$(dirname -- "$line")
"
        else
            skipped=$((skipped + 1))
        fi
    done < "$prev"

    # Sweep up directories those deletions just emptied — and only those. A
    # directory qualifies solely because it directly held a file WE shipped,
    # and plain `rmdir` (no -r, no -f) refuses to touch it if anything at all
    # is still inside, so nothing of the user's can be caught up in it.
    # Deepest first, so a nested pair collapses in a single pass.
    if [ -n "$dirs" ]; then
        printf '%s' "$dirs" \
            | LC_ALL=C sort -u \
            | awk -F/ '{ print NF "\t" $0 }' \
            | LC_ALL=C sort -rn \
            | cut -f2- \
            > "$prev.dirs" 2>/dev/null || true
        while IFS= read -r d; do
            [ -n "$d" ] || continue
            [ "$d" != "." ] || continue
            if ! manifest_path_ok "$d"; then continue; fi
            if manifest_protected "$d"; then continue; fi
            if [ ! -d "$d" ] || [ -L "$d" ]; then continue; fi
            if [ -d "$src/$d" ]; then continue; fi
            rmdir -- "$d" 2>/dev/null || true
        done < "$prev.dirs"
    fi

    if [ "$removed" -gt 0 ]; then
        echo "[update] Removed $removed file(s) this version no longer ships."
    else
        echo "[update] Nothing left over from the previous version."
    fi
    if [ "$skipped" -gt 0 ]; then
        echo "[update] Left $skipped path(s) alone (protected or unsafe to touch)."
    fi
}

# ZIP installs (no .git): download the latest ZIP and overlay it onto this
# folder. Settings, venv and .direct-mode aren't in the ZIP, so they survive.
# Runs inside main() (fully parsed), so rewriting update.sh mid-run is safe.
zip_update() {
    if [ -n "$PIN_VERSION" ]; then
        echo "[update] Downloading v$PIN_VERSION from GitHub…"
    else
        echo "[update] Downloading the latest version from GitHub…"
    fi
    local tmp="${TMPDIR:-/tmp}/fast-syncs-zip"
    rm -rf "$tmp" && mkdir -p "$tmp"
    if ! curl -fsSL -o "$tmp/repo.zip" "$ZIP_URL"; then
        # No release published yet (releases/latest 404s), or this machine
        # cannot reach it. Fall back to the branch ZIP so the button keeps
        # working; once the first release exists this branch is never taken.
        if [ -n "${FAST_SYNCS_ZIP_URL:-}" ] \
           || ! curl -fsSL -o "$tmp/repo.zip" "$FALLBACK_ZIP_URL"; then
            echo "[update] Could not download the update (offline, or the GitHub"
            echo "         repository is private / moved). Update manually instead:"
            echo "         re-download the ZIP and unzip it OVER this folder"
            echo "         (your settings and venv are kept)."
            NO_DL=1
            return 0
        fi
        echo "[update] No published release yet — used the latest code instead."
    fi
    if ! unzip -oq "$tmp/repo.zip" -d "$tmp"; then
        echo "[update] Could not unpack the update — update manually (re-download the ZIP)."
        NO_DL=1
        return 0
    fi
    # The ZIP contains one folder (fast-syncs-<branch>); copy its contents here.
    local src
    src=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [ -z "$src" ]; then
        echo "[update] Unexpected ZIP layout — update manually (re-download the ZIP)."
        NO_DL=1
        return 0
    fi
    # Save the manifest the PREVIOUS install left behind BEFORE the overlay
    # overwrites it — the difference between the two is the whole point.
    local prev_manifest="$tmp/previous-manifest"
    if [ -f "$MANIFEST" ]; then
        cp "$MANIFEST" "$prev_manifest" 2>/dev/null || rm -f "$prev_manifest"
    fi

    cp -R "$src/." .
    echo "[update] Files updated to the latest version."

    # Releases older than this feature carry no manifest; the overlay then
    # leaves the previous one untouched and there is nothing to compare, so
    # the cleanup simply does not run.
    if [ -f "$MANIFEST" ] && [ -f "$src/$MANIFEST" ]; then
        manifest_prune "$MANIFEST" "$prev_manifest" "$src"
    fi
}

# `--version X.Y.Z` (a leading "v" is fine) pins one published release —
# that is how the panel's "Roll back…" button installs an older version.
# Anything else on the command line is ignored, as before.
parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --version) PIN_VERSION="${2:-}"; shift; [ $# -gt 0 ] && shift ;;
            *) shift ;;
        esac
    done
    PIN_VERSION="${PIN_VERSION#v}"
    [ -n "$PIN_VERSION" ] || return 0
    # Digits and dots only: the value goes into a URL and a git ref.
    case "$PIN_VERSION" in
        *[!0-9.]*|.*)
            echo "[update] Not a version number: '$PIN_VERSION' (expected e.g. 0.28.9)."
            exit 1 ;;
    esac
    # The pinned release's own ZIP. Setting FAST_SYNCS_ZIP_URL too makes
    # zip_update skip its "latest code" fallback — installing some OTHER
    # version than the one asked for would defeat the point of picking one.
    ZIP_URL="https://github.com/aniruddhajd/fast-syncs/releases/download/v$PIN_VERSION/fast-syncs.zip"
    FAST_SYNCS_ZIP_URL="$ZIP_URL"
}

# git installs: move the checkout to the release's tag (detached HEAD). The
# branch we were on is remembered in .git/fast-syncs-branch so the next normal
# update can go back to it (git_leave_pin). Refuses — changes nothing — when
# local edits are in the way; it never overlays a ZIP over a git checkout here,
# because that would leave the tree dirty for every later pull.
git_pin() {
    local tag="v$PIN_VERSION" br
    echo "[update] Switching to $tag…"
    br=$(git symbolic-ref --short -q HEAD || true)
    if ! git fetch --quiet --tags origin; then
        echo "[update] Could not reach GitHub to fetch $tag — nothing was changed."
        NO_DL=1
        return 0
    fi
    if ! git -c advice.detachedHead=false checkout --quiet "$tag"; then
        echo "[update] Could not switch to $tag (no such release, or local edits"
        echo "         to tracked files are in the way) — nothing was changed."
        NO_DL=1
        return 0
    fi
    # Only on the FIRST roll back: a second one starts already detached and
    # must keep the original branch name.
    if [ -n "$br" ]; then echo "$br" > .git/fast-syncs-branch; fi
    echo "[update] Files switched to $tag."
}

# A normal update after a roll back: the checkout is on a detached tag, where
# `git pull` cannot work. Go back to the remembered branch (else production).
git_leave_pin() {
    git symbolic-ref -q HEAD >/dev/null && return 0
    local br="production"
    if [ -f .git/fast-syncs-branch ]; then br=$(cat .git/fast-syncs-branch); fi
    echo "[update] Leaving the rolled-back version — back to branch $br…"
    if git checkout --quiet "$br"; then rm -f .git/fast-syncs-branch; fi
    return 0
}

main() {
    cd "$(dirname "$0")"
    parse_args "$@"

    if [ -d .git ] && command -v git >/dev/null 2>&1 && [ -n "$PIN_VERSION" ]; then
        git_pin
    elif [ -d .git ] && command -v git >/dev/null 2>&1; then
        git_leave_pin
        echo "[update] Pulling latest changes…"
        # --ff-only aborts on a diverged / locally-modified checkout. Don't
        # let that fail the whole update (set -e) — fall back to the ZIP
        # overlay, which adds/overwrites tracked files without touching the
        # gitignored venv/settings. This is the common "update did nothing on
        # a git install" case.
        if ! git pull --ff-only; then
            echo "[update] git pull could not fast-forward (diverged or local"
            echo "         edits to tracked files) — falling back to ZIP overlay…"
            zip_update
        fi
    else
        zip_update
    fi

    # A venv whose base Python was uninstalled or upgraded (a Homebrew
    # upgrade breaks the symlinks) still exists on disk but cannot run.
    # Probe by executing; a broken venv is deleted and rebuilt through
    # setup.sh (this used to abort the whole update at the pip step).
    if [ -d venv ] && ! ./venv/bin/python3 -c 'import sys' >/dev/null 2>&1; then
        echo "[update] The venv is broken (its Python was removed or upgraded) — rebuilding…"
        rm -rf venv
    fi

    # setup.sh sets up the dubbing app itself, so the dubbing step below
    # can be skipped when it ran.
    SETUP_RAN=0
    if [ -d venv ]; then
        # Migrate installs made before the .direct-mode marker existed: if the
        # direct-mode libs are importable, this venv was set up with --direct.
        if [ ! -f .direct-mode ] && ./venv/bin/python3 -c "import google.genai" >/dev/null 2>&1; then
            touch .direct-mode
        fi
        echo "[update] Updating Python dependencies…"
        ./venv/bin/pip install --quiet --upgrade -r requirements.txt
        if [ -f .direct-mode ]; then
            echo "[update] Updating direct-mode dependencies…"
            ./venv/bin/pip install --quiet --upgrade -r requirements-direct.txt
        fi
    else
        echo "[update] No working venv — running setup.sh first."
        SETUP_RAN=1
        if [ -f .direct-mode ]; then bash setup.sh --direct; else bash setup.sh; fi
    fi

    # Bundled dubbing app (dubbing/): run its setup in automatic mode on
    # EVERY update — first update after the merge it creates dubbing/venv
    # and installs ffmpeg (this used to be a manual step people hit as
    # "venv not found" errors); later updates just refresh deps (fast).
    # Never fails the whole update. Skipped when setup.sh just ran — it
    # already did this itself.
    if [ "$SETUP_RAN" = "0" ] && [ -f dubbing/setup_mac.command ]; then
        echo "[update] Setting up / refreshing the dubbing app (first time can take a few minutes)…"
        bash dubbing/setup_mac.command --auto \
            || echo "[update] Dubbing setup reported a problem — run dubbing/setup_mac.command manually."
    fi

    echo
    if [ "$NO_DL" = "1" ]; then
        echo "Update finished, but the new version could NOT be downloaded —"
        echo "see the messages above. Your current version keeps working."
    elif [ -n "$PIN_VERSION" ]; then
        echo "Version $PIN_VERSION installed. Re-run the script in REAPER to use it."
        echo "(Update… in the panel's About pane brings you back to the newest.)"
    else
        echo "Update complete. Re-run the script in REAPER to use the new version."
    fi
}

# Sourcing this file with FAST_SYNCS_LIB=1 loads the helpers WITHOUT running
# an update — that is how the delete-safety logic above is tested. The `if` is
# a compound command, so bash still parses the whole thing before main runs:
# the protection against this file being rewritten mid-run (see the header)
# is unchanged.
if [ "${FAST_SYNCS_LIB:-}" != "1" ]; then main "$@"; exit $?; fi
