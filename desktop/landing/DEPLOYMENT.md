# Saddle AdPulse Landing Page - Deployment Guide

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Netlify       │     │    Supabase      │     │   SMTP Server   │
│  (Static Site)  │────▶│   (PostgreSQL)   │────▶│   (Email)       │
│  + Functions    │     │   + Webhooks     │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Setup Steps

### 1. Deploy to Netlify

1. Connect your GitHub repo to Netlify
2. Set build settings:
   - **Publish directory**: `landing` (or root if deploying just landing folder)
   - **Functions directory**: `netlify/functions`

### 2. Configure Netlify Environment Variables

Go to **Site Settings > Environment Variables** and add:

| Variable | Description | Example |
|----------|-------------|---------|
| `SMTP_HOST` | Your SMTP server | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP port | `587` |
| `SMTP_USER` | SMTP username | `your-email@gmail.com` |
| `SMTP_PASS` | SMTP password/app password | `xxxx-xxxx-xxxx-xxxx` |
| `SMTP_FROM` | From address for emails | `noreply@saddl.io` |
| `NOTIFICATION_EMAIL` | Where to send notifications | `aslam.yousuf@saddl.io` |
| `WEBHOOK_SECRET` | Secret to verify webhook calls | `your-random-secret-string` |

### 3. Update beta.html with Supabase Credentials

In `beta.html`, replace the placeholder values:

```javascript
const SUPABASE_URL = 'https://your-project.supabase.co';
const SUPABASE_ANON_KEY = 'your-anon-key-here';
```

> **Note**: The anon key is safe to expose - RLS policies protect your data.

### 4. Run the Database Migration

In Supabase SQL Editor, run the migration:

```sql
-- Run contents of migrations/create_beta_signups.sql
```

### 5. Configure Supabase Webhook

1. Go to **Supabase Dashboard > Database > Webhooks**
2. Create new webhook:
   - **Name**: `beta_signup_notification`
   - **Table**: `beta_signups`
   - **Events**: `INSERT`
   - **URL**: `https://your-site.netlify.app/.netlify/functions/beta-signup-notification`
   - **HTTP Headers**: 
     ```
     X-Webhook-Secret: your-random-secret-string
     ```

### 6. Test the Flow

1. Submit a test signup on the beta page
2. Check Supabase for the new row
3. Verify email notification received

## Files Created

| File | Purpose |
|------|---------|
| `netlify.toml` | Netlify build configuration |
| `package.json` | Dependencies (nodemailer) |
| `netlify/functions/beta-signup-notification.js` | Email notification function |
| `migrations/create_beta_signups.sql` | Database table + RLS policies |

## CORS Configuration

If you have CORS issues, add your Netlify domain to Supabase:
1. **Supabase Dashboard > Settings > API**
2. Add your domain to **Allowed Origins**

## Troubleshooting

### Form not saving to database
- Check browser console for errors
- Verify SUPABASE_URL and SUPABASE_ANON_KEY are correct
- Check Supabase RLS policies allow INSERT

### Not receiving email notifications
- Check Netlify function logs
- Verify SMTP credentials
- Check spam folder
- Verify webhook is configured correctly in Supabase
