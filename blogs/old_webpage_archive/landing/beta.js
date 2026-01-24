// Beta Signup Logic
// Version: 1.2 - Fixed: proper error handling, correct Supabase access

(function () {
    'use strict';

    console.log('[Beta] Script starting...');

    // Supabase configuration
    const SUPABASE_URL = 'https://wuakeiwxkjvhsnmkzywz.supabase.co';
    const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind1YWtlaXd4a2p2aHNubWt6eXd6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjYxMjE4MzYsImV4cCI6MjA4MTY5NzgzNn0.4n-RdlBEE-zGOxp3NsI8mKOcm10mEXUc9Fcz4-AyVe0';

    let supabaseClient = null;
    let initError = null;

    // Initialize Supabase when ready
    function initSupabase() {
        try {
            // Check various ways Supabase might be exposed
            if (window.supabase && typeof window.supabase.createClient === 'function') {
                supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
                console.log('[Beta] Supabase initialized successfully');
                return true;
            } else {
                console.warn('[Beta] window.supabase not available:', typeof window.supabase);
                initError = 'Supabase library not loaded. Ad blocker may be blocking it.';
                return false;
            }
        } catch (err) {
            console.error('[Beta] Supabase init error:', err);
            initError = err.message;
            return false;
        }
    }

    // Attach form handler
    function attachFormHandler() {
        const form = document.getElementById('betaSignupForm');

        if (!form) {
            console.error('[Beta] Form element not found!');
            return;
        }

        console.log('[Beta] Form found, attaching submit handler...');

        form.addEventListener('submit', async function (e) {
            // CRITICAL: Prevent default form submission
            e.preventDefault();
            e.stopPropagation();

            console.log('[Beta] Form submit intercepted');

            const submitBtn = form.querySelector('button[type="submit"]');
            const originalText = submitBtn ? submitBtn.textContent : '';

            if (submitBtn) {
                submitBtn.textContent = 'Submitting...';
                submitBtn.disabled = true;
            }

            // Check if Supabase is ready
            if (!supabaseClient) {
                // Try to init again in case it loaded late
                initSupabase();
            }

            if (!supabaseClient) {
                console.error('[Beta] Supabase not available for submission');
                alert('Unable to submit: Database connection failed.\n\n' +
                    (initError || 'Please try disabling ad blockers and refresh the page.'));
                if (submitBtn) {
                    submitBtn.textContent = originalText;
                    submitBtn.disabled = false;
                }
                return;
            }

            // Collect form data
            const formData = new FormData(form);
            const data = {
                name: formData.get('name'),
                email: formData.get('email'),
                role: formData.get('role'),
                accounts: formData.get('accounts'),
                monthly_spend: formData.get('spend'),
                goal: formData.get('goal') || null,
                source: 'landing_page'
            };

            console.log('[Beta] Submitting data:', { ...data, email: '***' });

            try {
                const { data: result, error } = await supabaseClient
                    .from('beta_signups')
                    .insert([data]);

                if (error) {
                    console.error('[Beta] Supabase error:', error);

                    if (error.code === '23505') {
                        alert('This email is already registered for beta access. We\'ll be in touch soon!');
                        if (submitBtn) {
                            submitBtn.textContent = originalText;
                            submitBtn.disabled = false;
                        }
                        return;
                    }

                    throw error;
                }

                console.log('[Beta] Signup successful!');

                // Show success message
                form.classList.add('hidden');
                const successMsg = document.getElementById('betaSuccessMessage');
                if (successMsg) {
                    successMsg.classList.remove('hidden');
                    successMsg.scrollIntoView({ behavior: 'smooth', block: 'center' });
                } else {
                    alert('Thanks! Your application has been received.');
                }

            } catch (err) {
                console.error('[Beta] Submission error:', err);
                alert('Something went wrong. Please try again.\n\nError: ' + (err.message || 'Unknown error'));
                if (submitBtn) {
                    submitBtn.textContent = originalText;
                    submitBtn.disabled = false;
                }
            }
        });

        console.log('[Beta] Submit handler attached successfully');
    }

    // Initialize when DOM is ready
    function init() {
        console.log('[Beta] Initializing...');
        initSupabase();
        attachFormHandler();
        console.log('[Beta] Initialization complete');
    }

    // Run init when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // DOM already loaded
        init();
    }

})();
