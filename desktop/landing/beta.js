// Beta Signup Logic
// Version: 1.2 - Enhanced Debugging & Robust Loading

(function () {
    console.log('[Beta] Script starting...');

    // Wait for DOM if it's not ready (though script at bottom of body usually means it is)
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initBeta);
    } else {
        initBeta();
    }

    function initBeta() {
        console.log('[Beta] Initializing...');

        // Supabase configuration
        const SUPABASE_URL = 'https://wuakeiwxkjvhsnmkzywz.supabase.co';
        const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind1YWtlaXd4a2p2aHNubWt6eXd6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjYxMjE4MzYsImV4cCI6MjA4MTY5NzgzNn0.4n-RdlBEE-zGOxp3NsI8mKOcm10mEXUc9Fcz4-AyVe0';

        let supabase = null;
        let supabaseLoadError = null;

        // Safely initialize Supabase
        try {
            // Check for UMD global 'supabase' (from window.supabase)
            if (typeof window.supabase !== 'undefined' && typeof window.supabase.createClient === 'function') {
                supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
                console.log('[Beta] Supabase initialized successfully');
            } else {
                console.warn('[Beta] Supabase JS not found on window object.');
                supabaseLoadError = 'Library did not load (possible ad blocker or CDN issue)';
            }
        } catch (err) {
            console.error('[Beta] Supabase init error:', err);
            supabaseLoadError = err.message;
        }

        const form = document.getElementById('betaSignupForm');
        if (!form) {
            console.error('[Beta] Beta signup form not found in DOM');
            return;
        }

        console.log('[Beta] Form found, attaching submit handler...');

        form.addEventListener('submit', async function (e) {
            // PREVENT DEFAULT is critical to stop URL change
            e.preventDefault();
            console.log('[Beta] Form submission intercepted');

            const submitBtn = this.querySelector('button[type="submit"]');
            const originalText = submitBtn.textContent;
            submitBtn.textContent = 'Submitting...';
            submitBtn.disabled = true;

            // If Supabase failed to load, alert immediately
            if (!supabase) {
                console.error('[Beta] Supabase not available');
                alert(`Unable to submit: The database connection failed. \nReason: ${supabaseLoadError || 'Unknown error'}.\n\nPlease try disabling ad blockers for this site.`);
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
                return;
            }

            const formData = new FormData(this);
            const data = {
                name: formData.get('name'),
                email: formData.get('email'),
                role: formData.get('role'),
                accounts: formData.get('accounts'),
                monthly_spend: formData.get('spend'),
                goal: formData.get('goal') || null,
                source: 'landing_page'
            };

            try {
                console.log('[Beta] Sending data to Supabase...', data);
                const { error } = await supabase.from('beta_signups').insert([data]);

                if (error) {
                    console.error('[Beta] Supabase insert error:', error);

                    // Handle duplicate email specifically
                    if (error.code === '23505') {
                        alert('This email is already registered for beta access. We\'ll be in touch soon!');
                    } else {
                        throw error;
                    }
                } else {
                    // Success path
                    console.log('[Beta] Signup successful');
                    this.classList.add('hidden');
                    const successMsg = document.getElementById('betaSuccessMessage');
                    if (successMsg) {
                        successMsg.classList.remove('hidden');
                        successMsg.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    } else {
                        alert('Thanks! You are on the list.');
                    }
                }
            } catch (error) {
                console.error('[Beta] Submission caught error:', error);
                alert(`Something went wrong. Please try again. \nError: ${error.message}`);
            } finally {
                // Restore button state if not successful (success hides form anyway)
                if (!this.classList.contains('hidden')) {
                    submitBtn.textContent = originalText;
                    submitBtn.disabled = false;
                }
            }
        });

        console.log('[Beta] Submit handler attached successfully');
        console.log('[Beta] Initialization complete');
    }
})();
