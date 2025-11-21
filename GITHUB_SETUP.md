# GitHub Setup Guide

Your repository is initialized and ready to push to GitHub!

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Repository name: `solar-predictive-maintenance` (or your preferred name)
4. Description: "Predictive maintenance models for solar assets using digital twin simulation"
5. Choose Public or Private
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

## Step 2: Connect and Push

After creating the repository, GitHub will show you commands. Use these:

```bash
cd /Users/peterhall/Desktop/Programmes/Tickets/solar-predictive-maintenance

# Add the remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/solar-predictive-maintenance.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/solar-predictive-maintenance.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
cd /Users/peterhall/Desktop/Programmes/Tickets/solar-predictive-maintenance
gh repo create solar-predictive-maintenance --public --source=. --remote=origin --push
```

## Verify

After pushing, visit your repository on GitHub to verify all files are there.

## Future Updates

To push future changes:

```bash
git add .
git commit -m "Your commit message"
git push
```

