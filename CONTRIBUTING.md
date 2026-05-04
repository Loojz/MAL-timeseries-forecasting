# Contributing Guidelines

> Best practices and rules for collaborating on the MAL Time Series Forecasting project. Inspired by the Gitflow workflow introduced in our lecture.

## Branching Strategy

We follow a simplified Gitflow model adapted to our 3-person team:

- `main` — stable, presentable version
- `develop` — integration branch (default)
- `feature/arima-btc` — Marvin's univariate analysis
- `feature/arima-eurusd` — Alper's univariate analysis
- `feature/arima-oil` — Luis' univariate analysis
- `feature/<shared-task>` — collaborative work

### Rules

- `main` is sacred. Only updated before presentations (04.05., 11.05., 18.05.)
- `develop` is the integration branch. All features merge here first
- Never push directly to `main` or `develop` — use Pull Requests
- One feature equals one branch. Keep branches focused and short-lived

### Branch Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Feature | feature/scope | feature/arima-oil |
| Fix | fix/scope | fix/data-loader-encoding |
| Documentation | docs/scope | docs/update-readme |
| Setup | setup/scope | setup/initial-repo-structure |

## Commit Message Convention

We follow the Conventional Commits standard. Format:

`type: short description in present tense`

Optionally followed by a body explaining the why, not the what.

### Types

- `feat` — new feature or analysis
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code restructuring without behavior change
- `test` — adding or updating tests
- `chore` — maintenance, dependencies, configuration

### Good Examples

- `feat: add ADF stationarity test to data pipeline`
- `fix: handle missing values in btc price series`
- `docs: extend README with setup instructions`

### Bad Examples

- `update`
- `fixed stuff`
- `asdf`
- `WIP`

### Rules of Thumb

- Use the imperative mood ("add", not "added" or "adds")
- Keep the subject line under 50 characters
- Capitalize the description, no period at the end
- Commit early and often — small, focused commits beat giant ones

## Pull Request Workflow

1. Create a branch from `develop`
2. Work locally, commit often
3. Push your branch to GitHub
4. Open a Pull Request targeting `develop`
5. Request review from at least one teammate
6. After approval and any requested changes, merge via GitHub
7. Delete the branch after merging

## Code Style

- Python: PEP 8 (use `black` formatter when in doubt)
- Notebooks: Clear narrative structure, with markdown explanations between code cells
- Comments: Explain why, not what. The code shows the what.
- Docstrings: Required for all functions in `src/`

## What NOT to Commit

- Raw data files larger than 50 MB (use Git LFS or external storage)
- API keys, passwords, credentials — use `.env` files (and add `.env` to `.gitignore`)
- IDE-specific files (`.idea/`, `.vscode/` — already in `.gitignore`)
- Notebook outputs with sensitive data
- `__pycache__/`, `.DS_Store`, virtual environments

## Resolving Merge Conflicts

If two team members edit the same file:

1. Pull the latest changes from develop into your branch
2. Git will mark conflict zones with conflict markers
3. Open the file, decide what to keep, remove the markers
4. Stage and commit the resolved file
5. Push the resolved branch

When in doubt: talk to your teammate before merging.

## Quick Reference for Team Members

### One-time setup

```
git clone git@github.com:Loojz/MAL-timeseries-forecasting.git
cd MAL-timeseries-forecasting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Starting a new feature

```
git checkout develop
git pull origin develop
git checkout -b feature/my-task
```

### Working

```
git add .
git commit -m "feat: short description of what I did"
git push -u origin feature/my-task
```

### Then on GitHub

- Open Pull Request targeting `develop`
- Add reviewers
- After at least one approval, merge and delete branch

### After merge — clean up locally

```
git checkout develop
git pull origin develop
git branch -d feature/my-task
```

## Communication

- Daily quick sync during class
- Issues on GitHub for tasks and bugs
- WhatsApp/Discord for fast questions

---

In case of fire: 1. git commit — 2. git push — 3. leave building.
