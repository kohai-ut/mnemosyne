#!/usr/bin/env bash
# Deploy Mnemosyne as a Hermes MemoryProvider via the plugin system.
# Works with both:
#   1. curl -sSL <url> | bash         (auto-clones the repo)
#   2. Running locally from the repo  (uses existing clone)
set -eo pipefail
# Note: set -u (nounset) is intentionally NOT used because BASH_SOURCE[0] is
# unset when piped via curl | bash. We use "${BASH_SOURCE[0]:-}" instead.

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
REPO_URL="https://github.com/AxDSan/mnemosyne.git"
REPO_CLONE_DIR="$HERMES_HOME/mnemosyne-repo"
TARGET_DIR="$HERMES_HOME/plugins/mnemosyne"

echo "🚀 Mnemosyne MemoryProvider Deploy"
echo "=================================="
echo ""

# --- Detect source: local repo clone, or piped via curl ---
_detect_source() {
    # Try BASH_SOURCE first (available when running from a file)
    if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        local provider_dir="$script_dir/hermes_memory_provider"
        if [ -d "$provider_dir" ]; then
            echo "$script_dir"
            return 0
        fi
    fi

    # Check existing persistent clone
    if [ -d "$REPO_CLONE_DIR/hermes_memory_provider" ]; then
        echo "📂 Found existing clone at $REPO_CLONE_DIR" >&2
        echo "$REPO_CLONE_DIR"
        return 0
    fi

    # Check if we're already in a mnemosyne repo (running from cwd)
    local cwd="${PWD}"
    if [ -d "$cwd/hermes_memory_provider" ]; then
        echo "📂 Using current directory: $cwd" >&2
        echo "$cwd"
        return 0
    fi

    # Need to clone
    echo "📦 Cloning Mnemosyne repository to $REPO_CLONE_DIR..." >&2
    git clone --depth 1 "$REPO_URL" "$REPO_CLONE_DIR" >/dev/null 2>&1
    echo "$REPO_CLONE_DIR"
}

REPO_ROOT="$(_detect_source)"
PROVIDER_DIR="$REPO_ROOT/hermes_memory_provider"

if [ ! -d "$PROVIDER_DIR" ]; then
    echo "❌ Error: hermes_memory_provider/ not found at $PROVIDER_DIR"
    echo "   Make sure the Mnemosyne repository is available."
    exit 1
fi

# Ensure plugins directory exists
mkdir -p "$HERMES_HOME/plugins"

# Remove existing symlink or directory
if [ -L "$TARGET_DIR" ]; then
    echo "🔄 Removing existing symlink: $TARGET_DIR"
    rm "$TARGET_DIR"
elif [ -d "$TARGET_DIR" ]; then
    echo "🔄 Removing existing directory: $TARGET_DIR"
    rm -rf "$TARGET_DIR"
fi

# Create symlink
ln -s "$PROVIDER_DIR" "$TARGET_DIR"
echo "✅ Symlinked: $TARGET_DIR -> $PROVIDER_DIR"

# Verify
if [ -L "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
    echo "✅ Deploy verified."
else
    echo "❌ Deploy failed."
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Set provider in config:"
echo "       hermes config set memory.provider mnemosyne"
echo ""
echo "  2. Or edit ~/.hermes/config.yaml:"
echo "       memory:"
echo "         provider: mnemosyne"
echo ""
echo "  3. Verify:"
echo "       hermes memory status"
echo "       hermes mnemosyne stats"
echo ""
