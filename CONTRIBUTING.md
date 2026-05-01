# Contributing

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production only. Never commit directly to main. |
| `dev` | Integration branch. All feature work merges here first. |
| `feature/*` | Individual work branches. Always branch off `dev`, never `main`. |
| `fix/*` | Bug fix branches. Same rules as `feature/*`. |

## Workflow

1. Pull latest `dev` before starting any work:
   ```bash
   git checkout dev && git pull origin dev
   ```

2. Create your branch off `dev`:
   ```bash
   git checkout -b feature/your-description
   ```

3. Do your work and commit regularly with clear messages.

4. Push and open a PR into `dev` — not `main`:
   ```bash
   git push origin feature/your-description
   ```

5. Noah reviews all PRs before merge.

6. `dev → main` PRs happen only when a feature set is ready to deploy.

## PR Requirements

- CI must pass (lint + tsc) before merge
- At least one review required
- No direct pushes to `main` or `dev`
