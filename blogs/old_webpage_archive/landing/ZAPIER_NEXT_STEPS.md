# Completing the Notification Loop

Right now, the flow is:
`Website Form` -> `Supabase Database` -> `Zapier Webhook`

The data stops at Zapier unless you tell it where to go next.

## Step 1: Catch the Data in Zapier
1.  Log in to **Zapier**.
2.  Open your Zap (the one with the URL you gave me).
3.  Click on the **Trigger** step ("Catch Hook").
4.  Click **"Test trigger"**.
    *   You should see the "Test from Saddle Agent" request I sent earlier.
    *   *Or* submit the live form on `saddl.io` and click "Find new records" to see real data.

## Step 2: Choose Where to Send It (The "Action")
You need to add a second step to your Zap to actually notify yourself.

### To Get an Email:
1.  Click **"+"** to add a step.
2.  Search for **"Gmail"** (or Outlook).
3.  Choose Event: **"Send Email"**.
4.  Connect your email account.
5.  **To:** `info@saddl.io` (or your personal email).
6.  **Subject:** `New Beta Signup: [Click and select 'Name' field]`
7.  **Body:**
    *   Name: `[Select 'Name' field]`
    *   Email: `[Select 'Email' field]`
    *   Spend: `[Select 'Monthly Spend' field]`

### To Get a Slack Message:
1.  Search for **"Slack"**.
2.  Event: **"Send Channel Message"**.
3.  Pick your internal `#leads` channel.
4.  Customize the message text with the fields from step 1.

## Step 3: Turn It On
1.  Click **Publish** in Zapier.
2.  That's it! Now every Supabase insert triggers this Zap, which sends the notification to your chosen destination.
