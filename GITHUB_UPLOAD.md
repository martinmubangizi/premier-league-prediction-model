# Uploading to GitHub, step by step

Written for someone who has not pushed a repo before. About 20 minutes.
Replace `YOUR-USERNAME` with your actual GitHub username everywhere it appears.

---

## Before you start

**1.** Unzip `pl-prediction-engine.zip`. You should see:

```
README.md  MODEL_CARD.md  LICENSE  .gitignore
pyproject.toml  requirements.txt  requirements-dev.txt
GITHUB_UPLOAD.md  LINKEDIN_GUIDE.md  scripts_download_data.sh
src/  tests/  tools/  docs/  output/  .github/workflows/
```

**1a.** Run the tests before you push anything. They need no data download:

```bash
pip install -r requirements-dev.txt
pytest
ruff check src tests
```

Expect 23 passing tests. If they pass locally they will pass in CI.

**2.** Check git is installed. Open a terminal and run:

```bash
git --version
```

If that errors, install it: `sudo apt install git` on Ubuntu, `brew install git`
on Mac, or download from [git-scm.com](https://git-scm.com/downloads) on Windows.

**3.** Tell git who you are (once per machine):

```bash
git config --global user.name "Martin Mubangizi"
git config --global user.email "mubangizimartin1@gmail.com"
```

---

## Step 1: Create the empty repo on GitHub

1. Go to [github.com/new](https://github.com/new)
2. **Repository name:** `premier-league-prediction-model`
3. **Description:** `A Premier League match prediction model validated against bookmaker odds`
4. Select **Public**
5. **Do not** tick "Add a README file", "Add .gitignore", or "Choose a license". You
   already have all three, and ticking them creates a conflict on your first push.
6. Click **Create repository**

Leave that page open. You will need the URL.

---

## Step 2: Set up authentication

GitHub stopped accepting passwords over HTTPS in 2021. You need a Personal Access
Token, which acts as your password when pushing.

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. **Generate new token** then **Generate new token (classic)**
3. **Note:** `laptop push access`
4. **Expiration:** 90 days
5. Tick the **`repo`** checkbox (this ticks all its children automatically)
6. Scroll down, **Generate token**
7. **Copy the token now.** It starts `ghp_`. GitHub will never show it again. Paste
   it somewhere safe for the next few minutes.

When git later asks for a password, paste this token, not your GitHub password.

---

## Step 3: Push the code

Open a terminal, navigate into the unzipped folder, and run these one at a time:

```bash
cd path/to/pl-prediction-engine

git init
git add .
git status
```

`git status` should list your files in green. Confirm that `data/raw/` is **not**
listed: `.gitignore` excludes it because the raw CSVs are ~45MB and are rebuilt by
script anyway.

```bash
git commit -m "Premier League prediction model: engine, backtest, dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/premier-league-prediction-model.git
git push -u origin main
```

On the last command git asks for credentials:
- **Username:** your GitHub username
- **Password:** paste the `ghp_...` token from Step 2

Refresh the GitHub page. Your files are there.

> **If it says `remote origin already exists`:** run
> `git remote set-url origin https://github.com/YOUR-USERNAME/premier-league-prediction-model.git`
> and push again.
>
> **If it says `failed to push some refs`:** you ticked one of the boxes in Step 1.
> Fix with `git pull origin main --allow-unrelated-histories` then push again.
>
> **If it says `src refspec main does not match any`:** the commit did not happen.
> Run `git add .` and `git commit -m "initial"` again, watching for errors.

---

## Step 4: Turn on the live dashboard

The `docs/` folder holds the dashboard as `index.html`. GitHub Pages serves it free.

1. On your repo page, click **Settings**
2. **Pages** in the left sidebar
3. Under "Build and deployment":
   - **Source:** Deploy from a branch
   - **Branch:** `main`, folder `/docs`
4. Click **Save**
5. Wait about two minutes, then reload the page. A green banner shows your URL:

```
https://YOUR-USERNAME.github.io/premier-league-prediction-model/
```

Open it on your phone to check it renders. Then go back to the repo home page,
click the gear icon next to **About**, and paste that URL into the **Website**
field. Add topics while you are there: `machine-learning`, `football-analytics`,
`sports-analytics`, `python`, `forecasting`.

---

## Step 5: Check it over before you share it

Walk through this list on the live URL:

- [ ] The "This is not a betting product" notice is the first thing you see
- [ ] All ten fixtures render, each with two bars
- [ ] The glossary explains Mdl and Mkt
- [ ] The technical appendix tables are readable, no overflow
- [ ] It works on your phone
- [ ] The footer says the project is not affiliated with the Premier League
- [ ] No em dashes anywhere (verified, but look anyway)
- [ ] Your name is in the LICENSE file
- [ ] The CI badge is green (Actions tab, after the first push)
- [ ] `MODEL_CARD.md` renders correctly on GitHub

Then read the README on GitHub top to bottom as a stranger would. That page is what
people judge the project by, more than the code.

---

## Updating it each week

The scheduled task regenerates the dashboard every Thursday. To publish a new one:

```bash
cd path/to/pl-prediction-engine
cp output/dashboard.html docs/index.html
git add docs/index.html output/
git commit -m "Week 2 forecasts"
git push
```

Pages redeploys in about a minute. Same URL.

If git asks for credentials every time and it gets tedious, run
`git config --global credential.helper store` once, enter the token one final
time, and it will be remembered.

---

## Adding the CI badge

After your first push, GitHub Actions runs lint and tests automatically. Add the
badge to the top of `README.md`, directly under the title:

```markdown
[![CI](https://github.com/YOUR-USERNAME/premier-league-prediction-model/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/premier-league-prediction-model/actions/workflows/ci.yml)
```

Commit and push it. A green badge on a public repo is worth more than most README
prose, because it is checkable.

---

## Two things worth doing later

**Protect the token.** If you ever paste it somewhere public, revoke it immediately
at [github.com/settings/tokens](https://github.com/settings/tokens). Tokens are
easy to regenerate.

**Tag a release** once matchweek 1 has been scored, so there is a fixed snapshot
people can point at:

```bash
git tag -a v1.0 -m "Week 1 forecasts, pre-results"
git push origin v1.0
```
