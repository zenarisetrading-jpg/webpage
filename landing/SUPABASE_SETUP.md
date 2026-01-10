# Final Step: Connect Beta Form to Zapier

Good news! I verified the Zapier webhook (`...ug3l5xp`) is active and receiving data.

To make the Beta Form send data to this webhook automatically, run **ONE** of the following options in your Supabase Dashboard.

## Option 1: The SQL Way (Quickest)
Copy and paste this code into the **SQL Editor** in Supabase and run it. This creates a trigger that fires every time a new row is added to `beta_signups`.

```sql
-- 1. Enable the HTTP extension if not already on
create extension if not exists "pg_net";

-- 2. Create the trigger function
create or replace function public.trigger_zapier_beta_signup()
returns trigger as $$
begin
  perform net.http_post(
      url:='https://hooks.zapier.com/hooks/catch/25994638/ug3l5xp/',
      body:=jsonb_build_object(
          'name', new.name,
          'email', new.email,
          'role', new.role,
          'accounts', new.accounts,
          'spend', new.monthly_spend,
          'goal', new.goal,
          'source', new.source,
          'timestamp', new.created_at
      )
  );
  return new;
end;
$$ language plpgsql;

-- 3. Attach it to the table
drop trigger if exists send_beta_signup_to_zapier on beta_signups;
create trigger send_beta_signup_to_zapier
  after insert on beta_signups
  for each row
  execute function public.trigger_zapier_beta_signup();
```

## Option 2: The UI Way (No Code)
1. Go to **Database** -> **Webhooks** in the Supabase sidebar.
2. Click **Create a new webhook**.
3. **Name**: `Send to Zapier`.
4. **Table**: `public.beta_signups`.
5. **Events**: Check `INSERT`.
6. **Type**: `HTTP Request`.
7. **Method**: `POST`.
8. **URL**: `https://hooks.zapier.com/hooks/catch/25994638/ug3l5xp/`.
9. Click **Confirm**.

## Testing
Once you've done either option:
1. Go to your live site (`saddl.io`).
2. Submit the Beta form.
3. Check your Zapier history—you should see the new lead appear instantly!
