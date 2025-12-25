"""
Authentication UI components for SADDLE.
"""
import streamlit as st
from auth.service import AuthService


# =============================================================================
# SIGN UP FORM
# =============================================================================
def render_signup_form() -> None:
    """Render a premium signup form with logo header."""
    import base64
    from pathlib import Path
    
    # Create centered layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Embossed Logo Header (same as login)
        logo_path = Path(__file__).parent.parent / "static" / "saddle_logo.png"
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
                <div style="
                    text-align: center;
                    margin-top: -2rem;
                    margin-bottom: 1.5rem;
                ">
                    <div style="
                        display: inline-block;
                        padding: 20px 32px 16px 32px;
                        background: linear-gradient(145deg, rgba(91, 86, 112, 0.15) 0%, rgba(11, 11, 13, 0.4) 100%);
                        border-radius: 20px;
                        border: 1px solid rgba(91, 86, 112, 0.3);
                        box-shadow: 
                            inset 0 2px 4px rgba(255, 255, 255, 0.05),
                            inset 0 -2px 4px rgba(0, 0, 0, 0.3),
                            0 8px 32px rgba(0, 0, 0, 0.4);
                    ">
                        <img src="data:image/png;base64,{logo_data}" style="height: 120px;" />
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<p style="color: #E9EAF0; font-size: 1.5rem; font-weight: 700; text-align: center; margin-bottom: 1rem;">Create Account</p>', unsafe_allow_html=True)
        
        with st.form("signup_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@company.com")
            password = st.text_input("Password", type="password", placeholder="Minimum 8 characters")
            confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            
            if submitted:
                if not email or not password:
                    st.error("Please fill in all fields")
                elif password != confirm:
                    st.error("Passwords do not match")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    auth = AuthService()
                    result = auth.sign_up(email, password)
                    if result["success"]:
                        st.success(result.get("message", "Account created! Check your email."))
                    else:
                        st.error(result.get("error", "Signup failed"))
        
        # Back to Login - SAME SIZE as Create Account (inside col2)
        if st.button("← Back to Login", key="back_to_login", type="primary", use_container_width=True):
            st.session_state['auth_view'] = 'login'
            st.rerun()


# =============================================================================
# LOGIN FORM - PREMIUM UI (Brand Aligned)
# =============================================================================
def render_login_form() -> dict:
    """Render a premium login form matching SADDLE brand guidelines."""
    auth = AuthService()
    
    # Brand-aligned Login CSS
    st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Full page brand background */
    .stApp {
        background: linear-gradient(180deg, #0B0B0D 0%, #1a1825 50%, #5B5670 100%) !important;
    }
    
    .login-logo {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    .login-title {
        color: #E9EAF0;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        font-family: Inter, -apple-system, sans-serif;
    }
    
    .login-subtitle {
        color: #9A9AAA;
        text-align: center;
        margin-bottom: 2.5rem;
        font-size: 1rem;
    }
    
    /* Input styling - brand aligned */
    .stTextInput > div > div > input {
        background: rgba(11, 11, 13, 0.7) !important;
        border: 1px solid rgba(91, 86, 112, 0.4) !important;
        border-radius: 12px !important;
        color: #E9EAF0 !important;
        padding: 1rem 1.2rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2A8EC9 !important;
        box-shadow: 0 0 0 3px rgba(42, 142, 201, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #6B6B7B !important;
    }
    
    /* PRIMARY CTA - Embossed Saddle Purple with Signal Blue accent */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #5B5670 0%, #464156 100%) !important;
        color: #E9EAF0 !important;
        border: 1px solid rgba(233, 234, 240, 0.15) !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.03em !important;
        box-shadow: 
            0 4px 20px rgba(91, 86, 112, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
        text-transform: none !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6c6684 0%, #5B5670 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 
            0 8px 30px rgba(91, 86, 112, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.15),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Form styling */
    [data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* Labels - Slate Grey */
    .stTextInput label {
        color: #9A9AAA !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.4rem !important;
    }
    
    /* Checkbox styling */
    .stCheckbox label {
        color: #E9EAF0 !important;
    }
    
    /* Links - Signal Blue */
    a {
        color: #2A8EC9 !important;
    }
    a:hover {
        color: #8FC9D6 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered login layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Logo - 2x SIZE (240px) with EMBOSSED CONTAINER
        import base64
        from pathlib import Path
        logo_path = Path(__file__).parent.parent / "static" / "saddle_logo.png"
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
                <div style="
                    text-align: center;
                    margin-top: -2rem;
                    margin-bottom: 1.5rem;
                ">
                    <div style="
                        display: inline-block;
                        padding: 24px 40px 20px 40px;
                        background: linear-gradient(145deg, rgba(91, 86, 112, 0.15) 0%, rgba(11, 11, 13, 0.4) 100%);
                        border-radius: 24px;
                        border: 1px solid rgba(91, 86, 112, 0.3);
                        box-shadow: 
                            inset 0 2px 4px rgba(255, 255, 255, 0.05),
                            inset 0 -2px 4px rgba(0, 0, 0, 0.3),
                            0 8px 32px rgba(0, 0, 0, 0.4);
                    ">
                        <img src="data:image/png;base64,{logo_data}" style="height: 200px;" />
                        <p style="
                            color: #8FC9D6;
                            font-size: 0.9rem;
                            font-weight: 500;
                            letter-spacing: 0.05em;
                            margin-top: 8px;
                            margin-bottom: 0;
                        ">Amazon advertising, simplified</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<p class="login-title" style="margin-bottom: 4px;">Welcome Back</p>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle" style="margin-bottom: 1rem;">Sign in to your account</p>', unsafe_allow_html=True)
        
        # Login form - COMPACT: Email & Password on same row
        with st.form("login_form", clear_on_submit=False):
            col_email, col_pwd = st.columns(2)
            with col_email:
                email = st.text_input("Email", placeholder="you@company.com", key="login_email", label_visibility="collapsed")
            with col_pwd:
                password = st.text_input("Password", type="password", placeholder="Password", key="login_password", label_visibility="collapsed")
            
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            
            if submitted:
                if not email or not password:
                    st.error("Please enter your email and password")
                    return {"authenticated": False}
                
                result = auth.sign_in(email, password)
                if result["success"]:
                    st.success("Welcome back!")
                    st.rerun()
                else:
                    st.error(result.get("error", "Login failed"))
        
        # Forgot Password / Sign up Links
        col_fp, col_su = st.columns(2)
        with col_fp:
            if st.button("Forgot password?", key="forgot_pwd_btn", type="tertiary", use_container_width=True):
                st.session_state['auth_view'] = 'reset'
                st.rerun()
        with col_su:
            if st.button("Sign up", key="signup_btn", type="tertiary", use_container_width=True):
                st.session_state['auth_view'] = 'signup'
                st.rerun()
    
    return {"authenticated": auth.is_authenticated()}


# =============================================================================
# PASSWORD RESET
# =============================================================================
def render_password_reset() -> None:
    """Render password reset form."""
    st.markdown('<p style="color: #E9EAF0; font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;">Reset Password</p>', unsafe_allow_html=True)
    
    with st.form("reset_form"):
        email = st.text_input("Email", placeholder="Enter your email address")
        submitted = st.form_submit_button("Send Reset Link", type="primary", use_container_width=True)
        
        if submitted:
            if not email:
                st.error("Please enter your email")
            else:
                auth = AuthService()
                result = auth.reset_password(email)
                if result["success"]:
                    st.success(result.get("message", "Check your email for reset link"))
                else:
                    st.error(result.get("error", "Failed to send reset email"))


# =============================================================================
# USER MENU - TOP RIGHT HEADER
# =============================================================================
def render_user_menu() -> None:
    """
    Render a top-right header bar with user info.
    Account selection and logout are in sidebar for reliability.
    """
    auth = AuthService()
    user_email = auth.get_user_email()
    user = auth.get_current_user()
    
    if not user_email:
        return
    
    # Get user display name from metadata
    user_metadata = getattr(user, 'user_metadata', {}) or {}
    display_name = user_metadata.get('full_name', user_email.split('@')[0])
    
    # Get theme mode for styling
    theme_mode = st.session_state.get('theme_mode', 'dark')
    
    if theme_mode == 'dark':
        header_bg = "rgba(22, 22, 35, 0.95)"
        header_border = "rgba(91, 85, 111, 0.3)"
        name_color = "#E9EAF0"
        email_color = "#9A9AAA"
    else:
        header_bg = "rgba(255, 255, 255, 0.95)"
        header_border = "rgba(221, 217, 212, 0.8)"
        name_color = "#1A1D24"
        email_color = "#4A4F5C"
    
    # Get initials for avatar
    initials = ''.join([n[0].upper() for n in display_name.split()[:2]]) if display_name else user_email[0].upper()
    
    # Get current account
    current_account = st.session_state.get('active_account_name', '')
    
    # Fixed header CSS and HTML
    st.markdown(f"""
    <style>
    .top-header {{
        position: fixed;
        top: 60px;
        right: 24px;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 14px;
        background: {header_bg};
        border: 1px solid {header_border};
        border-radius: 12px;
        padding: 10px 18px;
        backdrop-filter: blur(20px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }}
    .header-account-badge {{
        background: rgba(42, 142, 201, 0.12);
        border: 1px solid rgba(42, 142, 201, 0.25);
        border-radius: 6px;
        padding: 6px 12px;
        color: #2A8EC9;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .header-divider {{
        width: 1px;
        height: 28px;
        background: rgba(154, 154, 170, 0.2);
    }}
    .header-user {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .header-avatar {{
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #5B5670 0%, #464156 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #E9EAF0;
        font-size: 0.9rem;
        font-weight: 600;
    }}
    .header-name {{
        color: {name_color};
        font-size: 0.9rem;
        font-weight: 600;
    }}
    .header-email {{
        color: {email_color};
        font-size: 0.75rem;
    }}
    </style>
    
    <div class="top-header">
        <span class="header-account-badge">{current_account if current_account else 'No Account'}</span>
        <div class="header-divider"></div>
        <div class="header-user">
            <div class="header-avatar">{initials}</div>
            <div>
                <div class="header-name">{display_name}</div>
                <div class="header-email">{user_email}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# AUTH PAGE - ENTRY POINT FOR UNAUTHENTICATED USERS  
# =============================================================================
def render_auth_page() -> None:
    """
    Render the authentication page for unauthenticated users.
    Routes between login, signup, and password reset based on auth_view state.
    """
    # Initialize auth_view if not set
    if 'auth_view' not in st.session_state:
        st.session_state['auth_view'] = 'login'
    
    view = st.session_state.get('auth_view', 'login')
    
    if view == 'signup':
        render_signup_form()
        # Back to Login button is now inside render_signup_form()
    elif view == 'reset':
        render_password_reset()
        # Back to login - LARGE BUTTON
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Login", key="back_to_login_reset", type="secondary", use_container_width=True):
            st.session_state['auth_view'] = 'login'
            st.rerun()
    else:
        render_login_form()

