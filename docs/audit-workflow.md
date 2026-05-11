# Mnemosyne Docs Audit — Reusable Workflow Checklist

**Purpose:** Bi-weekly cross-reference audit of docs site against codebase.
**Cadence:** Every 2 weeks (or after any version bump / major feature merge).
**Audience:** Hermes Agent (send this document as context to re-run the audit).

---

## Pre-Flight (5 min)

- [ ] Confirm working directory: `cd /root/.hermes/projects/mnemosyne`
- [ ] Check current version: `python -c "import mnemosyne; print(mnemosyne.__version__)"`
- [ ] Check last audit: `cat /root/.hermes/projects/mnemosyne-docs/.planning/AUDIT-REPORT-*.md | head -5`
- [ ] Note: All patches go to `content/` files, then mirror to `src/app/(docs)/` copies.

---

## Phase 1: Codebase Surface Map (15 min)

Generate a fresh codebase map. This tells you what actually exists.

**Method:** Delegate to a subagent with:
```
Map the entire Mnemosyne codebase surface in /root/.hermes/projects/mnemosyne.
Catalog every class, method signature, CLI command, configuration option,
tool schema (MCP + Hermes plugin), API endpoint, and importer.
Output structured JSON to mnemosyne_codebase_surface.json.
```

**Key files to check for changes since last audit:**
- `mnemosyne/core/beam.py` — BeamMemory (new methods, changed defaults)
- `mnemosyne/core/memory.py` — Mnemosyne class (new methods, changed signatures)
- `mnemosyne/mcp_tools.py` — MCP tool definitions (new tools, changed params)
- `mnemosyne/core/importers/` — New import providers
- Plugin yamls — `plugin.yaml`, `hermes_plugin/plugin.yaml` — tool counts, hook names
- `pyproject.toml` — dependencies, entry points

**Verify key numbers from code:**
- [ ] Default `top_k` in `recall()` — check `beam.py` line with `def recall`
- [ ] MCP tool count — check `mcp_tools.py` for `get_tool_definitions()`
- [ ] Hermes plugin tool count — check plugin.yaml `tools:` field
- [ ] Hook names — check `hermes_plugin/__init__.py` for hook registration
- [ ] Config env vars — check `mnemosyne/core/beam.py` and `memory.py` for `os.getenv`
- [ ] config.yaml keys — check `hermes_plugin/__init__.py` for config reads

---

## Phase 2: Critical Page Audit (30 min)

These 10 pages are the ones most likely to rot. Check them every time.

### 1. `getting-started/configuration.mdx`
- [ ] Every env var listed EXISTS in the actual codebase (grep for `os.getenv`)
- [ ] Defaults match code defaults
- [ ] Config file is correctly `config.yaml` (not `mnemosyne.yaml`)
- [ ] Class name is `Mnemosyne` (not `Memory`)
- [ ] Embedding model matches `MNEMOSYNE_EMBEDDING_MODEL` default
- [ ] config.yaml keys match actual `memory.mnemosyne.*` structure

### 2. `api/python-sdk.mdx`
- [ ] Constructor signature matches `Mnemosyne.__init__` exactly
- [ ] `recall()` default `top_k` matches code default
- [ ] All methods listed in docs exist in code (grep for `def method_name`)
- [ ] All public methods from code are documented in docs
- [ ] V2 Properties table matches actual properties on Mnemosyne class
- [ ] Stream API methods match `MemoryStream` class
- [ ] DeltaSync methods match `DeltaSync` class
- [ ] Hermes Plugin Tools table lists ALL tools from plugin.yaml

### 3. `api/tool-schema.mdx`
- [ ] Every tool definition's `required` params match MCP tool code
- [ ] Every tool definition's `properties` match MCP tool code
- [ ] No fictional parameters (like `tags` that was here before)
- [ ] Number of tools matches actual tool count

### 4. `api/hermes-plugin.mdx`
- [ ] Hooks table names match actual hook registration in `hermes_plugin/__init__.py`
- [ ] Hook descriptions are accurate
- [ ] Tool list matches plugin.yaml
- [ ] No fictional configuration options

### 5. `api/mcp-server.mdx`
- [ ] Tool names match `mcp_tools.py` exactly
- [ ] Tool count matches
- [ ] Transport options (stdio, SSE) match CLI

### 6. `architecture/beam-overview.mdx`
- [ ] Number of memory tiers is correct (3: working, episodic, scratchpad)
- [ ] TripleStore is correctly described as separate, not a 4th tier
- [ ] Capacity numbers match env var defaults
- [ ] Latency claims match benchmark data

### 7. `architecture/system-design.mdx`
- [ ] No fictional components (like the "REST API" box that was here)
- [ ] Component names match actual classes/modules
- [ ] Technology stack table is accurate

### 8. `operations/performance.mdx`
- [ ] Memory tier names match actual tables
- [ ] Memory usage numbers are realistic
- [ ] No fictional tiers (like "Semantic Memory")
- [ ] Benchmark numbers are recent

### 9. `architecture/streaming.mdx`
- [ ] API method names match actual `MemoryStream` class
- [ ] API method names match actual `DeltaSync` class
- [ ] `compute_delta()` shows required `peer_id` param

### 10. `architecture/plugin-system.mdx`
- [ ] Registration method names match `PluginManager` class
- [ ] Hook signatures match `MnemosynePlugin` abstract class
- [ ] Built-in plugin names match actual plugins

---

## Phase 3: Comparison Pages (10 min)

These should match current version and feature set.

- [ ] `comparisons/*.mdx` — all say v2.5.0 (or current version)
- [ ] `comparisons/*.mdx` — "Last updated" dates are recent
- [ ] Tool counts referenced match actual counts
- [ ] Provider counts match actual importers list

### Comparison pages:
- [ ] `comparisons/honcho.mdx`
- [ ] `comparisons/zep.mdx`
- [ ] `comparisons/mem0.mdx`
- [ ] `comparisons/letta.mdx`
- [ ] `comparisons/cognee.mdx`
- [ ] `comparisons/supermemory.mdx`
- [ ] `comparisons/hindsight.mdx`

---

## Phase 4: Landing/Quick-Start Pages (5 min)

- [ ] `getting-started/quick-start.mdx` — version number, code snippets work
- [ ] `getting-started/installation.mdx` — pip install command correct
- [ ] `getting-started/first-steps.mdx` — API usage matches current signatures
- [ ] `migration/overview.mdx` — provider count accurate, version current

---

## Phase 5: Fix and Commit (15-30 min)

### Fixing approach:
1. **Use patch tool** for targeted edits — never rewrite entire files with sed/read_file
2. **Fix `content/` files first**, then mirror to `src/app/(docs)/` copies
3. **Verify with:** `grep -rn 'old_string' content/ src/` before declaring done

### Mirroring script:
```python
import os, shutil
content_dir = "/root/.hermes/projects/mnemosyne-docs/content"
app_dir = "/root/.hermes/projects/mnemosyne-docs/src/app/(docs)"
for rel_path in modified_files:
    src = os.path.join(content_dir, rel_path)
    dir_part = os.path.dirname(rel_path)
    name_part = os.path.splitext(os.path.basename(rel_path))[0]
    dst = os.path.join(app_dir, dir_part, name_part, "page.mdx")
    shutil.copy2(src, dst)
```

### Commit template:
```
fix(docs): bi-weekly audit — [brief summary of what changed]
```

---

## Phase 6: Website Cross-Check (5 min)

- [ ] `mnemosyne-website/src/components/HomePage.tsx` — BEAM labels still correct
- [ ] `mnemosyne-website/src/data/changelog.json` — last sync date is recent
- [ ] Website version matches codebase version

---

## Phase 7: Report (5 min)

- [ ] Write executive report to `.planning/AUDIT-REPORT-YYYY-MM-DD.md`
- [ ] Include: pages audited, issues found, fixes applied, remaining risks
- [ ] Update this workflow if new pain points discovered

---

## Pain Points Log (Lessons Learned)

### From May 11, 2026 Audit:

1. **Never use sed with pipe characters in search strings.** The `|` in markdown tables conflicts with sed's `|` delimiter. Use Python's `str.replace()` or the `patch` tool instead.

2. **The patch tool is the safest edit method.** It does fuzzy matching and won't corrupt files. The `sed` command can silently fail or corrupt files when special characters are involved.

3. **Mirror files are a trap.** `content/` and `src/app/(docs)/` are separate copies. If you only fix one, the other remains stale. Always sync both. Better yet: fix the build system to use one source of truth.

4. **The configuration page was the worst rot.** It had zero correspondence with actual code. This happened because config systems are the hardest to keep in sync — they have no type checking and vary across environments.

5. **Subagent timeouts on large scans.** When auditing 67 pages, a single subagent timed out at 600s. Break the work into chunks: one subagent for codebase mapping, separate subagents for page groups.

6. **Don't assume documentation is accurate.** Some pages were clearly generated from assumptions. Always verify against source code, not other documentation.

7. **subagent `read_file` can drop data.** When reading files with `read_file` and rewriting with `write_file`, frontmatter/export blocks can be lost. Use the `patch` tool for all edits, or verify content integrity after writes.

---

*To re-run this audit: send this document to Hermes Agent with the message "Run the bi-weekly docs audit using AUDIT-WORKFLOW.md"*
