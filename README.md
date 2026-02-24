# COMP4702 Coursework (UQ Machine Learning, S1 2026)

Coursework repository for `COMP4702`, organised for reproducible study, practicals, and assignment experiments.

## What Is In This Repo

- `lectures/`: weekly lecture notes/notebooks (Jupytext paired where used)
- `pracs/`: practical materials, PDFs, and worked notebooks
- `assignment/`: report/code/notebooks/figures/submission workspace
- `references/`: exams, course summary table, textbook, and course reference files
- `shared/`: reusable fixtures/templates/scripts for small experiments
- `scripts/`: repo utility scripts (notebook execution, Jupytext sync, requirements sync, smoke test)

Current populated examples include:
- lecture materials in `lectures/week-01` to `lectures/week-13` (some weeks are placeholders/in-progress)
- a worked Week 2 practical notebook triple in `pracs/week-02/` (`.ipynb`, `.md`, `.py`)

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
scripts/run_notebook.sh pracs/week-02/week-02-prac.ipynb
```

If no path is given, it defaults to `pracs/week-02/week-02-prac.ipynb`.

## Jupytext Triple Sync Workflow (`.ipynb` + `.md` + `.py`)

This repo is configured for Jupytext pairing via `.jupytext.toml`:
- formats: `ipynb,md,py:percent`

Set up pairing (first time for a notebook stem):

```bash
scripts/setup_notebook_workflow.sh lectures/week-05/week-05-lecture.ipynb
```

Sync after editing any one of the paired files:

```bash
scripts/sync_notebook.sh pracs/week-02/week-02-prac.md
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

A repo-local pre-commit hook (`.githooks/pre-commit`) auto-syncs `requirements.txt` when `environment.yml` changes. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

## Public Repo Export / Publish (not relevent for the public repo)

This repo can export a curated public copy to a separate git repository (for sharing notes/materials without the full private repo contents).

Export only (sync files into the public copy):

```bash
./scripts/export_public.sh
```

Export + stage + commit + push the public repo:

```bash
./scripts/publish_public.sh
```

Optional arguments:

```bash
./scripts/publish_public.sh /home/user/Documents/Uni/COMP4702-public-copy "Update public notes"
```

If you already ran `export_public.sh` manually and the public repo is now dirty, you can skip the export step and publish the existing changes:

```bash
./scripts/publish_public.sh --skip-export --allow-dirty /home/user/Documents/Uni/COMP4702-public-copy "Update public notes"
```

Notes:
- `scripts/publish_public.sh` refuses to run if the public repo has uncommitted changes (safety check).
- Use `--skip-export --allow-dirty` only when you intentionally want to publish changes already present in the public repo working tree.
- The destination public repo must already exist, be a git repo, and have an `origin` remote configured.

## Notebook Output Hygiene

- `.gitattributes` configures `nbstripout` for `*.ipynb`
- notebook diffs are configured with `diff=ipynb`

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

- Prefer `conda run -n comp4702 ...` for non-interactive Python commands
- Keep datasets and generated outputs out of git unless intentionally small fixtures
- Use fixed seeds and reproducible notebook/script workflows
- Do not change evaluation protocols silently

See `AGENTS.md` for the full project workflow and guardrails used in this repo.
