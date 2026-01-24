# Supabase Setup for Beta Signups

## 1. Database Schema
Run the following SQL in your Supabase SQL Editor to create the `beta_signups` table:

```sql
CREATE TABLE IF NOT EXISTS beta_signups (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    role VARCHAR(50),
    accounts VARCHAR(20),
    monthly_spend VARCHAR(50),
    goal TEXT,
    source VARCHAR(50) DEFAULT 'landing_page',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending'
);

-- Enable RLS
ALTER TABLE beta_signups ENABLE ROW LEVEL SECURITY;

-- Allow anonymous inserts (for the public form)
CREATE POLICY "Allow anonymous insert" ON beta_signups FOR INSERT WITH CHECK (true);

-- Allow authenticated users (you) to read
CREATE POLICY "Authenticated users can read" ON beta_signups FOR SELECT USING (auth.role() = 'authenticated');
```

## 2. Connect the Form
In `desktop/landing/beta.html`, update line 145 with your **Supabase Anon Key**:

```javascript
const SUPABASE_ANON_KEY = 'YOUR_KEY_HERE';
```

You can find this key in Supabase Dashboard -> Project Settings -> API.

## 3. Email Notification (New Submission)
To receive an email at `info@saddl.io` when a new user signs up, the best way is to use a **Supabase Edge Function** or a **Database Webhook**.

### Option A: Database Webhook (Easiest if you use Zapier/Make)
1.  Go to Supabase Dashboard -> Database -> Webhooks.
2.  Create a new webhook.
3.  Name: `beta-signup-notification`.
4.  Table: `public.beta_signups`.
5.  Events: `INSERT`.
6.  Type: `HTTP Request`.
7.  URL: Your Zapier/Make webhook URL (which sends the email).

### Option B: Supabase Edge Function (Developer Way)
1.  Install Supabase CLI.
2.  Run `supabase functions new beta-notification`.
3.  Add code to send email (using Resend, SendGrid, or SMTP).
4.  Deploy with `supabase functions deploy beta-notification`.
5.  Create a database trigger to call this function on insert.

**Example Database Trigger for Edge Function:**

```sql
create trigger "on_beta_signup"
  after insert on public.beta_signups
  for each row execute function supabase_functions.http_request(
    'https://<project-ref>.supabase.co/functions/v1/beta-notification',
    'POST',
    '{"Content-type":"application/json"}',
    '{}',
    '1000'
  );
```
