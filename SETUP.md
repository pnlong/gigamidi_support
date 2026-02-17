# Environment setup and moving to another computer

## Export your current mamba environment (on this computer)

**Portable** (recommended; only records packages you installed; dependencies are resolved on the new machine):

```bash
mamba activate gigamidi
mamba env export --from-history > environment.exported.yml
```

**Exact** (full lockfile; use only if the other computer has the same OS and architecture):

```bash
mamba activate gigamidi
mamba env export > environment.exported.yml
```

Copy `environment.exported.yml` (and the repo) to the other computer.

## On the new computer

**If you have an exported file:**

```bash
cd /path/to/gigamidi
mamba env create -f environment.exported.yml
mamba activate gigamidi
```

**If you use the repo’s `environment.yml`:**

```bash
cd /path/to/gigamidi
mamba env create -f environment.yml
mamba activate gigamidi
pip install -r emotion_genre/requirements.txt
```

**Using the setup script (if present):**

```bash
cd /path/to/gigamidi
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh create    # create env
./scripts/setup_env.sh export    # print export instructions
```

## One-liner to create env from repo (no export file)

```bash
mamba create -n gigamidi python=3.10 -y && mamba activate gigamidi && pip install -r emotion_genre/requirements.txt
```
