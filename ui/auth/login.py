"""
Login Screen
============
Minimal login interface wiring.
PRD Reference: ORG_USERS_ROLES_PRD.md §12

Note: This is a UI shell. Integration with auth middleware happens in the main app flow.
"""

import streamlit as st
from core.auth.middleware import AuthError



def render_forgot_password():
    """Renders the forgot password form."""
    
    # 1. Logo
    import base64
    from pathlib import Path
    logo_path = Path(__file__).parent.parent.parent / "static" / "saddle_logo.png"
    
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 0.5rem;">
                <img src="data:image/png;base64,{logo_data}" style="width: 320px; max-width: 80%; opacity: 1.0;"> 
            </div>
            """, 
            unsafe_allow_html=True
        )

    # 2. Header
    st.markdown('<h1 class="login-title">Reset Password</h1>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Enter your email to receive reset instructions</div>', unsafe_allow_html=True)

    # 3. Form
    with st.form("forgot_form"):
        email = st.text_input("Email", placeholder="user@example.com")
        
        submitted = st.form_submit_button("Send Instructions", type="primary")
        
        if submitted:
            if not email:
                st.error("Please enter your email address")
            else:
                from core.auth.service import AuthService
                service = AuthService()
                if service.request_password_reset(email):
                    st.success("If an account exists for this email, instructions have been sent.")
                else:
                    st.error("Unable to process request.")

    # 4. Back Link
    if st.button("← Back to Log In", type="secondary", use_container_width=True):
        st.session_state['auth_view'] = 'login'
        st.rerun()

def render_login():
    """Renders the login flow (Dispatcher)."""
    
    # State Management
    if 'auth_view' not in st.session_state:
        st.session_state['auth_view'] = 'login'

    # Global CSS for Auth Screens
    st.markdown("""
        <style>
        /* Widened container for better desktop experience */
        .block-container { max-width: 600px; padding-top: 2rem; padding-bottom: 5rem; }
        
        /* Ensure image is perfectly centered */
        img { display: block; margin-left: auto; margin-right: auto; }
        
        h1.login-title { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 2.2rem; text-align: center; margin-top: 0; margin-bottom: 0.5rem; color: white; }
        div.login-subtitle { text-align: center; color: #94a3b8; margin-bottom: 2.5rem; font-size: 1rem; }
        div[data-testid="stForm"] { border: 1px solid rgba(255, 255, 255, 0.1); background-color: rgba(255, 255, 255, 0.02); padding: 3rem; border-radius: 16px; }
        button[kind="primary"] { width: 100%; background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important; border: none; color: white !important; font-weight: 600; padding: 0.75rem 0; font-size: 1rem; }
        button[kind="primary"]:hover { box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4); transform: translateY(-1px); }
        /* Link Button Styling */
        button[kind="secondary"] { border: none !important; background: transparent !important; color: #94a3b8 !important; }
        button[kind="secondary"]:hover { color: white !important; }
        #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    if st.session_state['auth_view'] == 'forgot_password':
        render_forgot_password()
        return

    # --- LOGIN VIEW ---
    
    # 1. Logo
    import base64
    from pathlib import Path
    logo_path = Path(__file__).parent.parent.parent / "static" / "saddle_logo.png"
    
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 0.5rem;">
                <img src="data:image/png;base64,{logo_data}" style="width: 320px; max-width: 80%; opacity: 1.0;"> 
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.title("SADDLE")

    # 2. Header
    st.markdown('<h1 class="login-title">Welcome Back</h1>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Sign in to your account</div>', unsafe_allow_html=True)
    
    # 3. Form
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="user@example.com")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        submitted = st.form_submit_button("Log In", type="primary")
        
        if submitted:
            if not email or not password:
                st.error("Please enter email and password")
            else:
                from core.auth.service import AuthService
                auth = AuthService()
                
                result = auth.sign_in(email, password)
                if result["success"]:
                    st.success(f"Welcome back!")
                    st.rerun()
                else:
                    st.error(result.get("error", "Login failed"))

    # 4. Footer / Forgot Pwd
    col1, col2 = st.columns([1, 1])
    with col1: pass
    with col2: pass
        
    if st.button("Forgot Password?", type="secondary", use_container_width=True):
        st.session_state['auth_view'] = 'forgot_password'
        st.rerun()

    st.markdown("""
        <div style="text-align: center; margin-top: 0.5rem; font-size: 0.85rem;">
            <span style="color: #94a3b8;">New to Saddle?</span> 
            <a href="#" style="color: white; font-weight: 600; text-decoration: none;">Create an account</a>
        </div>
    """, unsafe_allow_html=True)
