# Git workflow on macOS and Windows

This repository uses `.gitattributes` and `.editorconfig` so text files are
stored consistently with UTF-8 and LF line endings on both platforms.

## First setup on each computer

```bash
git clone https://github.com/1jsjs/furiosa-practice.git
cd furiosa-practice
git config user.name "Your Name"
git config user.email "you@example.com"
```

GitHub authentication is configured separately on each computer. Do not copy
passwords, personal access tokens, or SSH private keys into this repository.

## Start working

```bash
git switch main
git pull --ff-only origin main
```

## Save and share work

```bash
git status
git add <files>
git commit -m "Describe the change"
git push origin main
```

Pull before starting work on the other computer. Commit or stash local changes
before pulling if Git reports that the working tree is not clean.
