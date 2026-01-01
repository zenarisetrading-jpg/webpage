"""
Authentication UI components for SADDLE.
"""
import streamlit as st
from auth.service import AuthService


# =============================================================================
# SHARED STYLES FOR AUTH PAGES
# =============================================================================
def _inject_auth_styles():
    """Inject shared CSS for auth pages."""
    st.markdown("""
    <style>
    /* Hide Streamlit chrome and prevent scrolling */
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background: #0B0B0D !important;
        overflow: hidden !important;
    }
    
    /* Force full viewport without scroll */
    html, body, [data-testid="stAppViewContainer"], .main {
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    .main .block-container {
        padding: 0 !important; 
        max-width: 100% !important;
        margin: 0 !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* Remove column gaps */
    [data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 100vh !important;
    }
    
    [data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* Premium Input styling - compact */
    .stTextInput > div > div > input {
        background: rgba(91, 86, 112, 0.08) !important;
        border: 1px solid rgba(91, 86, 112, 0.25) !important;
        border-radius: 10px !important;
        color: #E9EAF0 !important;
        padding: 12px 16px !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #5B5670 !important;
        box-shadow: 0 0 0 2px rgba(91, 86, 112, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(154, 154, 170, 0.6) !important;
    }
    .stTextInput label {
        color: #9A9AAA !important; 
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    .stTextInput {margin-bottom: 8px !important;}
    
    /* Premium Button styling */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #5B5670 0%, #4a4560 100%) !important;
        color: #E9EAF0 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(91, 86, 112, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #6c6684 0%, #5B5670 100%) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button[kind="tertiary"],
    .stButton > button[kind="secondary"] {
        color: #9A9AAA !important;
        background: transparent !important;
        border: 1px solid rgba(91, 86, 112, 0.25) !important;
        border-radius: 10px !important;
        font-size: 0.8rem !important;
        padding: 10px !important;
    }
    .stButton > button[kind="tertiary"]:hover,
    .stButton > button[kind="secondary"]:hover {
        color: #E9EAF0 !important;
        border-color: #5B5670 !important;
        background: rgba(91, 86, 112, 0.1) !important;
    }
    
    [data-testid="stForm"] {border: none !important; padding: 0 !important;}
    
    /* Divider */
    .auth-divider {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 16px 0;
        color: #5B5670;
        font-size: 0.75rem;
    }
    .auth-divider::before, .auth-divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(91, 86, 112, 0.25);
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SIGN UP FORM - 50:50 SPLIT DESIGN
# =============================================================================
def render_signup_form() -> None:
    """Render premium signup form with 50:50 split layout."""
    import base64
    from pathlib import Path
    
    # Load assets
    logo_path = Path(__file__).parent.parent / "static" / "saddle_logo.png"
    bg_path = Path(__file__).parent.parent / "assets" / "login_bg.png"
    
    logo_b64 = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    
    bg_b64 = ""
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
    
    # Inject CSS
    st.markdown(f"""
    <style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .stApp {{background: #0B0B0D !important;}}
    .block-container {{padding: 0 !important; max-width: 100% !important; margin: 0 !important;}}
    [data-testid="stHorizontalBlock"] {{gap: 0 !important; margin: 0 !important; padding: 0 !important;}}
    [data-testid="column"] {{padding: 0 !important; margin: 0 !important;}}
    
    .stTextInput > div > div > input {{
        background: #0B0B0D !important;
        border: 1px solid rgba(91, 86, 112, 0.4) !important;
        border-radius: 6px !important;
        color: #E9EAF0 !important;
        padding: 10px 12px !important;
        font-size: 0.85rem !important;
    }}
    .stTextInput label {{color: #9A9AAA !important; font-size: 0.75rem !important;}}
    .stFormSubmitButton > button {{
        background: #5B5670 !important;
        color: #E9EAF0 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px !important;
        font-weight: 600 !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # 50:50 columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        if bg_b64:
            st.markdown(f'''
            <div style="height: 100vh; background: url(data:image/png;base64,{bg_b64}) center/cover no-repeat;"></div>
            ''', unsafe_allow_html=True)
    
    with col_right:
        # Vertical centering
        st.markdown('<div style="height: 8vh;"></div>', unsafe_allow_html=True)

        # Center the form content
        _, form_col, _ = st.columns([0.2, 0.6, 0.2])

        with form_col:
            # Logo with tagline
            if logo_b64:
                st.markdown(f'''
                <div style="text-align: center; margin-bottom: 40px;">
                    <img src="data:image/png;base64,{logo_b64}" style="height: 180px; margin-bottom: 8px;" />
                    <p style="color: #9A9AAA; font-size: 0.85rem; margin: 0; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif; text-transform: uppercase; letter-spacing: 1.2px;">Amazon advertising, simplified</p>
                </div>
                ''', unsafe_allow_html=True)

            # Signup heading with subtitle
            st.markdown('''
            <div style="margin-bottom: 40px;">
                <h1 style="color: #E9EAF0; font-size: 2.2rem; font-weight: 700; margin: 0 0 8px 0; letter-spacing: -0.5px;">Create Account</h1>
                <p style="color: #9A9AAA; font-size: 0.95rem; margin: 0; font-weight: 400;">Sign up to get started with AdPulse</p>
            </div>
            ''', unsafe_allow_html=True)

            # Enhanced input styling (same as login)
            st.markdown('''
            <style>
            .stTextInput > div > div > input {
                background: rgba(91, 86, 112, 0.12) !important;
                border: 1px solid rgba(91, 86, 112, 0.25) !important;
                border-radius: 8px !important;
                color: #E9EAF0 !important;
                padding: 14px 16px !important;
                font-size: 0.95rem !important;
            }
            .stTextInput > div > div > input:focus {
                border-color: #5B5670 !important;
                box-shadow: 0 0 0 2px rgba(91, 86, 112, 0.15) !important;
            }
            .stTextInput > div > div > input::placeholder {
                color: rgba(154, 154, 170, 0.5) !important;
            }
            .stTextInput label {
                color: #9A9AAA !important;
                font-size: 0.85rem !important;
                font-weight: 500 !important;
                margin-bottom: 8px !important;
            }
            .stTextInput {margin-bottom: 20px !important;}

            /* Premium gradient button */
            .stFormSubmitButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 14px 24px !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
                transition: all 0.3s ease !important;
            }
            .stFormSubmitButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            }

            /* Secondary button for Sign in */
            .stButton > button[kind="secondary"] {
                background: transparent !important;
                border: 1px solid rgba(91, 86, 112, 0.35) !important;
                color: #E9EAF0 !important;
                border-radius: 8px !important;
                padding: 12px 24px !important;
                font-weight: 600 !important;
                font-size: 0.95rem !important;
            }
            .stButton > button[kind="secondary"]:hover {
                border-color: #667eea !important;
                background: rgba(102, 126, 234, 0.1) !important;
            }

            [data-testid="stForm"] {border: none !important; padding: 0 !important;}
            </style>
            ''', unsafe_allow_html=True)

            # Signup Form
            with st.form("signup_form", clear_on_submit=False):
                email = st.text_input("Email address", placeholder="name@example.com")
                password = st.text_input("Password", type="password", placeholder="Minimum 8 characters")
                confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm password")

                st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

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

            # Already have account - larger button
            st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center; margin-bottom: 12px;">', unsafe_allow_html=True)
            st.markdown('<span style="color: #9A9AAA; font-size: 0.9rem;">Already have an account?</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Sign in", key="back_to_login", type="secondary", use_container_width=True):
                st.session_state['auth_view'] = 'login'
                st.rerun()


# =============================================================================
# LOGIN FORM - PREMIUM 50:50 SPLIT DESIGN
# =============================================================================
def render_login_form() -> dict:
    """Render premium login with 50:50 split layout."""
    auth = AuthService()
    
    import base64
    from pathlib import Path
    
    # Load assets
    logo_path = Path(__file__).parent.parent / "static" / "saddle_logo.png"
    bg_path = Path(__file__).parent.parent / "assets" / "login_bg.png"
    
    logo_b64 = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    
    bg_b64 = ""
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
    
    # Inject CSS
    st.markdown(f"""
    <style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .stApp {{background: #0B0B0D !important;}}
    
    .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
        margin: 0 !important;
    }}
    
    [data-testid="stHorizontalBlock"] {{
        gap: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    
    [data-testid="column"] {{
        padding: 0 !important;
        margin: 0 !important;
    }}
    
    .stTextInput > div > div > input {{
        background: rgba(91, 86, 112, 0.1) !important;
        border: 1px solid rgba(91, 86, 112, 0.3) !important;
        border-radius: 10px !important;
        color: #E9EAF0 !important;
        padding: 12px 16px !important;
    }}
    .stTextInput label {{color: #9A9AAA !important; font-size: 0.85rem !important;}}
    .stTextInput {{margin-bottom: 8px !important;}}
    
    .stFormSubmitButton > button, .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #5B5670 0%, #4a4560 100%) !important;
        color: #E9EAF0 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-weight: 600 !important;
    }}
    
    .stButton > button[kind="tertiary"], .stButton > button[kind="secondary"] {{
        background: transparent !important;
        border: 1px solid rgba(91, 86, 112, 0.3) !important;
        color: #9A9AAA !important;
        border-radius: 10px !important;
        font-size: 0.85rem !important;
    }}
    
    [data-testid="stForm"] {{border: none !important; padding: 0 !important;}}
    </style>
    """, unsafe_allow_html=True)
    
    # 50:50 columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Background image panel
        if bg_b64:
            st.markdown(f'''
            <div style="
                height: 100vh;
                background: url(data:image/png;base64,{bg_b64}) center/cover no-repeat;
            "></div>
            ''', unsafe_allow_html=True)
    
    with col_right:
        # Vertical centering
        st.markdown('<div style="height: 8vh;"></div>', unsafe_allow_html=True)

        # Center the form content with more padding
        _, form_col, _ = st.columns([0.2, 0.6, 0.2])

        with form_col:
            # Logo with tagline
            if logo_b64:
                st.markdown(f'''
                <div style="text-align: center; margin-bottom: 40px;">
                    <img src="data:image/png;base64,{logo_b64}" style="height: 180px; margin-bottom: 8px;" />
                    <p style="color: #9A9AAA; font-size: 0.85rem; margin: 0; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif; text-transform: uppercase; letter-spacing: 1.2px;">Amazon advertising, simplified</p>
                </div>
                ''', unsafe_allow_html=True)

            # Login heading with subtitle
            st.markdown('''
            <div style="margin-bottom: 40px;">
                <h1 style="color: #E9EAF0; font-size: 2.2rem; font-weight: 700; margin: 0 0 8px 0; letter-spacing: -0.5px;">Login</h1>
                <p style="color: #9A9AAA; font-size: 0.95rem; margin: 0; font-weight: 400;">Sign in to your account to continue</p>
            </div>
            ''', unsafe_allow_html=True)

            # Enhanced input styling
            st.markdown('''
            <style>
            .stTextInput > div > div > input {
                background: rgba(91, 86, 112, 0.12) !important;
                border: 1px solid rgba(91, 86, 112, 0.25) !important;
                border-radius: 8px !important;
                color: #E9EAF0 !important;
                padding: 14px 16px !important;
                font-size: 0.95rem !important;
            }
            .stTextInput > div > div > input:focus {
                border-color: #5B5670 !important;
                box-shadow: 0 0 0 2px rgba(91, 86, 112, 0.15) !important;
            }
            .stTextInput > div > div > input::placeholder {
                color: rgba(154, 154, 170, 0.5) !important;
            }
            .stTextInput label {
                color: #9A9AAA !important;
                font-size: 0.85rem !important;
                font-weight: 500 !important;
                margin-bottom: 8px !important;
            }
            .stTextInput {margin-bottom: 20px !important;}

            /* Premium gradient button */
            .stFormSubmitButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 14px 24px !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
                transition: all 0.3s ease !important;
            }
            .stFormSubmitButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            }

            /* Tertiary button for links */
            .stButton > button[kind="tertiary"] {
                background: transparent !important;
                border: none !important;
                color: #667eea !important;
                padding: 0 !important;
                font-size: 0.9rem !important;
                font-weight: 500 !important;
                text-decoration: none !important;
            }
            .stButton > button[kind="tertiary"]:hover {
                color: #764ba2 !important;
                text-decoration: underline !important;
            }

            /* Align bottom row elements */
            div[data-testid="column"] .stButton {
                display: inline-block !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            div[data-testid="column"] span {
                display: inline !important;
                vertical-align: middle !important;
            }

            [data-testid="stForm"] {border: none !important; padding: 0 !important;}
            </style>
            ''', unsafe_allow_html=True)

            # Login Form
            with st.form("login_form", clear_on_submit=False):
                # Email field
                email = st.text_input("Email address", placeholder="name@example.com", key="login_email")

                # Password field
                password = st.text_input("Password", type="password", placeholder="password", key="login_password")

                st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

                if submitted:
                    if not email or not password:
                        st.error("Please enter email & password")
                        return {"authenticated": False}
                    result = auth.sign_in(email, password)
                    if result["success"]:
                        st.success("Welcome back!")
                        st.rerun()
                    else:
                        st.error(result.get("error", "Login failed"))

            # Bottom row - clean single-line layout
            st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)

            col_left_link, col_right_link = st.columns(2)

            with col_left_link:
                # Text and button on same line
                st.markdown('<span style="color: #9A9AAA; font-size: 0.9rem;">Not registered? </span>', unsafe_allow_html=True)
                if st.button("Create account", key="create_acc", type="tertiary"):
                    st.session_state['auth_view'] = 'signup'
                    st.rerun()

            with col_right_link:
                # Right-aligned button
                st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
                if st.button("Forgot password?", key="forgot_pwd_bottom", type="tertiary"):
                    st.session_state['auth_view'] = 'reset'
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    return {"authenticated": auth.is_authenticated()}


# =============================================================================
# PASSWORD RESET - 50:50 SPLIT DESIGN
# =============================================================================
def render_password_reset() -> None:
    """Render password reset form with 50:50 split layout."""
    import base64
    from pathlib import Path
    
    # Load assets
    logo_path = Path(__file__).parent.parent / "static" / "saddle_logo.png"
    bg_path = Path(__file__).parent.parent / "assets" / "login_bg.png"
    
    logo_b64 = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    
    bg_b64 = ""
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
    
    # Inject CSS
    st.markdown(f"""
    <style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .stApp {{background: #0B0B0D !important;}}
    .block-container {{padding: 0 !important; max-width: 100% !important; margin: 0 !important;}}
    [data-testid="stHorizontalBlock"] {{gap: 0 !important; margin: 0 !important; padding: 0 !important;}}
    [data-testid="column"] {{padding: 0 !important; margin: 0 !important;}}
    
    .stTextInput > div > div > input {{
        background: #0B0B0D !important;
        border: 1px solid rgba(91, 86, 112, 0.4) !important;
        border-radius: 6px !important;
        color: #E9EAF0 !important;
        padding: 10px 12px !important;
        font-size: 0.85rem !important;
    }}
    .stTextInput label {{color: #9A9AAA !important; font-size: 0.75rem !important;}}
    .stFormSubmitButton > button {{
        background: #5B5670 !important;
        color: #E9EAF0 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px !important;
        font-weight: 600 !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # 50:50 columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        if bg_b64:
            st.markdown(f'''
            <div style="height: 100vh; background: url(data:image/png;base64,{bg_b64}) center/cover no-repeat;"></div>
            ''', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        _, form_col, _ = st.columns([0.15, 0.7, 0.15])
        
        with form_col:
            # Logo with tagline
            if logo_b64:
                st.markdown(f'''
                <div style="text-align: center; margin-bottom: 24px;">
                    <img src="data:image/png;base64,{logo_b64}" style="height: 180px; margin-bottom: -10px;" />
                    <p style="color: #9A9AAA; font-size: 0.85rem; margin: 0; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Amazon advertising, simplified</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Reset Card Container
            st.markdown('''
            <div style="
                background: rgba(91, 86, 112, 0.15);
                border: 1px solid rgba(91, 86, 112, 0.4);
                border-radius: 12px;
                padding: 24px;
            ">
                <h3 style="color: #E9EAF0; font-size: 1.2rem; font-weight: 600; margin: 0 0 20px 0;">Reset Password</h3>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('<div style="margin-top: -60px; padding: 0 24px 24px 24px; background: rgba(91, 86, 112, 0.15); border-radius: 0 0 12px 12px; border: 1px solid rgba(91, 86, 112, 0.4); border-top: none;">', unsafe_allow_html=True)
            
            # Form
            with st.form("reset_form"):
                email = st.text_input("Email address", placeholder="Enter your email address")
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
            
            # Back to login link
            st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
            st.markdown('<span style="color: #9A9AAA; font-size: 0.8rem;">Remember your password? </span>', unsafe_allow_html=True)
            if st.button("Sign in", key="back_to_login_reset", type="tertiary"):
                st.session_state['auth_view'] = 'login'
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)


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

