# Supabase Edge Function for Email Notifications

To automatically send an email to `info@saddl.io` when a new user requests beta access, you should use a Supabase Edge Function.

## Prerequisites
1.  **Resend API Key**: Create an account at [resend.com](https://resend.com) and get an API key (it's the easiest way to send emails from edge functions).
2.  **Supabase CLI**: Installed on your machine.

## 1. Create the Function

Run this command in your terminal (root of your project):

```bash
supabase functions new beta-notification
```

## 2. Add Code

Replace the content of `supabase/functions/beta-notification/index.ts` with:

```typescript
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const RESEND_API_KEY = Deno.env.get("RESEND_API_KEY");

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const { record } = await req.json();

    if (!record || !record.email) {
      throw new Error("Missing record or email");
    }

    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${RESEND_API_KEY}`,
      },
      body: JSON.stringify({
        from: "AdPulse Beta <onboarding@resend.dev>", // Or your verified domain
        to: ["info@saddl.io"],
        subject: `New Beta Request: ${record.name}`,
        html: `
          <h1>New Beta Access Request</h1>
          <p><strong>Name:</strong> ${record.name}</p>
          <p><strong>Email:</strong> ${record.email}</p>
          <p><strong>Role:</strong> ${record.role}</p>
          <p><strong>Accounts:</strong> ${record.accounts}</p>
          <p><strong>Spend:</strong> ${record.monthly_spend}</p>
          <p><strong>Goal:</strong> ${record.goal || "Not specified"}</p>
          <br>
          <p>Login to Supabase to manage approval.</p>
        `,
      }),
    });

    const data = await res.json();
    return new Response(JSON.stringify(data), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });

  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 400,
    });
  }
});
```

## 3. Deploy

```bash
supabase functions deploy beta-notification
supabase secrets set RESEND_API_KEY=re_123456789 --no-verify-jwt
```

## 4. Create Database Trigger

Run this SQL in your Supabase Dashboard to trigger the email on every new insert:

```sql
create trigger "on_beta_signup_email"
  after insert on public.beta_signups
  for each row execute function supabase_functions.http_request(
    'https://wuakeiwxkjvhsnmkzywz.supabase.co/functions/v1/beta-notification',
    'POST',
    '{"Content-type":"application/json"}',
    '{}',
    '1000'
  );
```
*(Note: You might need to enable the `pg_net` extension for `http_request` or use the UI to add the webhook if you prefer).*

## Alternative: Webhook (Simpler)

1.  Go to Supabase Dashboard > Database > Webhooks.
2.  Enable Webhook for `INSERT` on `beta_signups`.
3.  Point it to a Zapier or Make.com webhook URL (which then sends the email).
