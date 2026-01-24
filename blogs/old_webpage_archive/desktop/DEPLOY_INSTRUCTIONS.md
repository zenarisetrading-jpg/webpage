# Production Deployment Guide

## 1. Prerequisites
Since Git is not installed on your system, you will need to install it to deploy to Streamlit Cloud.
- **Download Git**: [https://git-scm.com/download/win](https://git-scm.com/download/win)
- Install it and select "Use Git from the Windows Command Prompt" during setup.

## 2. Push to GitHub
Once Git is installed, open your terminal (Command Prompt or PowerShell) in this folder and run:

```bash
git init
git add .
git commit -m "Initial commit for production"
# (You will need to create a new repository on GitHub.com first)
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## 3. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Log in with GitHub.
3. Click **"New App"**.
4. Select your repository (`YOUR_REPO_NAME`) and the main file (`ppcsuite_v4.py`).
5. **IMPORTANT**: Click **"Advanced Settings"** -> **"Secrets"**.
6. Paste your Database URL:
   ```toml
   DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@db.YOUR_PROJECT.supabase.co:5432/postgres"
   ```
7. Click **"Deploy"**.

## 4. Environment Variables
- **DATABASE_URL**: Connection string for your Supabase/PostgreSQL database.
  - Format: `postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres`
