# COMP4702 Coursework (UQ Machine Learning, S1 2026)

Coursework repository for `COMP4702`, organised for reproducible study, practicals, and assignment experiments.

<!-- AUTO:PROJECT_OVERVIEW:START -->
## Project Review Snapshot (auto-generated)

- `lectures/` is populated with 12 week folder(s) and 26 non-placeholder file(s).
- `pracs/` (not public) is populated with 12 week folder(s) and 27 non-placeholder file(s).
- `references/` is populated with 22 non-placeholder file(s), including 6 top-level PDF reference(s).
- `assignment/` (not public) is currently scaffolded (`.gitkeep`/folders only).
- `shared/` (not public) is currently scaffolded (`.gitkeep`/folders only).
- `figures/` (not public) is currently empty.
- `2025 Material/` (not public) is populated with 4 non-placeholder file(s).

## Repo Layout (With Public Visibility)

Top-level paths/files currently in use:

- `.codex/` (not public): local Codex skill/config files.
- `.githooks/` (not public): local repo hook(s), including pre-commit automation.
- `.jupyter_tmp/` (not public): local runtime/temp Jupyter state.
- `2025 Material/` (not public): prior-year notes/slides references.
- `assignment/` (not public): assignment report/code/notebook workspace.
- `figures/` (not public): local generated figures workspace.
- `lectures/`: weekly lecture notes/notebooks/assets.
- `pracs/` (not public): practical materials, datasets, and worked notebooks.
- `references/`: official exam/course/textbook reference materials.
- `scripts/`: repo utility scripts; `scripts/add_user_preference.sh` (not public), `scripts/export_public.sh` (not public), `scripts/publish_public.sh` (not public), `scripts/sync_project_docs.py` (not public) are private-repo-only.
- `shared/` (not public): reusable local fixtures/templates/scripts.
- `AGENTS.md` (not public): local coding-agent workflow instructions.
- `.gitattributes` (not public): notebook clean/diff settings.
- `.gitignore` (not public): local ignore rules.
- `.jupytext.toml` (not public): Jupytext pairing config.
- `environment.yml`: conda environment source-of-truth.
- `README.md`: repository documentation.
- `requirements.txt`: pip export generated from the conda environment history.
- `.claude/` (not public): project directory.

## Public Sync Scope

`./scripts/export_public.sh` (not public) syncs only this curated subset to the public copy:

- `lectures/` (excluding `*.mp4` lecture recordings (not public))
- `references/`
- `scripts/` (excluding `scripts/add_user_preference.sh` (not public), `scripts/export_public.sh` (not public), `scripts/publish_public.sh` (not public), `scripts/sync_project_docs.py` (not public))
- `environment.yml`
- `README.md`
- `requirements.txt`

Everything outside that allowlist should be treated as `(not public)`.
<!-- AUTO:PROJECT_OVERVIEW:END -->

## Setup (Conda Environment)

`environment.yml` is the source of truth.

```bash
conda env create -f environment.yml
conda activate comp4702
```

Quick environment check:

```bash
conda run -n comp4702 python -c "import sys; print(sys.executable)"
```

## Running Notebooks (Recommended)

Use the repo script to execute notebooks in the correct environment and avoid `~/.ipython` permission issues:

```bash
scripts/run_notebook.sh lectures/week-03/week-03-lecture.ipynb
scripts/run_notebook.sh pracs/week-02/week-02-prac.ipynb  # (not public)
```

If no path is given, it defaults to `pracs/week-02/week-02-prac.ipynb` (not public).

## Jupytext Triple Sync Workflow (`.ipynb` + `.md` + `.py`)

This repo is configured for Jupytext pairing via `.jupytext.toml` (not public):

- formats: `ipynb,md,py:percent`

Set up pairing (first time for a notebook stem):

```bash
scripts/setup_notebook_workflow.sh lectures/week-05/week-05-lecture.ipynb
```

Sync after editing any one of the paired files:

```bash
scripts/sync_notebook.sh pracs/week-02/week-02-prac.md  # (not public)
```

Both scripts run `jupytext` via `conda run -n comp4702`.

Optional shell aliases:

```bash
alias nbsetup='/home/user/Documents/Uni/COMP4702/scripts/setup_notebook_workflow.sh'
alias nbsync='/home/user/Documents/Uni/COMP4702/scripts/sync_notebook.sh'
```

## Dependency Sync (`requirements.txt`)

`requirements.txt` is generated from the `comp4702` conda environment history.

Manual sync:

```bash
./scripts/sync_requirements.sh
```

A repo-local pre-commit hook (`.githooks/pre-commit`) (not public) auto-syncs managed docs (`README.md`, `AGENTS.md`) and also syncs `requirements.txt` when `environment.yml` changes. Enable it once per clone:

```bash
git config core.hooksPath .githooks  # (not public)
```

## Documentation Sync Automation

Manual sync:

```bash
conda run -n comp4702 python scripts/sync_project_docs.py
```

Check-only mode (fails if docs are stale):

```bash
conda run -n comp4702 python scripts/sync_project_docs.py --check
```

Add a durable user preference and re-sync AGENTS:

```bash
scripts/add_user_preference.sh "Preference text"  # (not public preference source: .codex/user-preferences.md)
```

## Public Repo Export / Publish (Private Repo Workflow)

This private repo can export a curated public copy to a separate git repository.

Export only (sync files into the public copy):

```bash
./scripts/export_public.sh  # (not public)
```

Export + stage + commit + push the public repo:

```bash
./scripts/publish_public.sh  # (not public)
```

Optional arguments:

```bash
./scripts/publish_public.sh /home/user/Documents/Uni/COMP4702-public-copy "Update public notes"  # (not public)
```

If you already ran `export_public.sh` (not public) manually and the public repo is now dirty, you can skip the export step and publish the existing changes:

```bash
./scripts/publish_public.sh --skip-export --allow-dirty /home/user/Documents/Uni/COMP4702-public-copy "Update public notes"  # (not public)
```

Notes:

- `scripts/publish_public.sh` (not public) refuses to run if the public repo has uncommitted changes (safety check).
- Use `--skip-export --allow-dirty` only when you intentionally want to publish changes already present in the public repo working tree.
- The destination public repo must already exist, be a git repo, and have an `origin` remote configured.

## Notebook Output Hygiene

- `.gitattributes` (not public) configures `nbstripout` for `*.ipynb`.
- notebook diffs are configured with `diff=ipynb`.

If needed, (re)install `nbstripout` into this repo:

```bash
conda run -n comp4702 nbstripout --install --attributes .gitattributes
```

## Smoke Test

Run a minimal import/train/plot smoke test in the course environment:

```bash
conda run -n comp4702 python scripts/smoke_test.py
```

This checks basic functionality for `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

## Repo Conventions (Short Version)

- Prefer `conda run -n comp4702 ...` for non-interactive Python commands.
- Keep datasets and generated outputs out of git unless intentionally small fixtures.
- Use fixed seeds and reproducible notebook/script workflows.
- Do not change evaluation protocols silently.

See `AGENTS.md` (not public) for the full project workflow and guardrails used in this repo.
