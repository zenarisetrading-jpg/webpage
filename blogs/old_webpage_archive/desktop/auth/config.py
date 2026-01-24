"""
Supabase client configuration with dual environment support.
Reads from st.secrets (Streamlit Cloud) first, falls back to .env (local development).
"""
import os
import streamlit as st
from supabase import create_client, Client


def get_env_var(key: str) -> str:
    """
    Get environment variable from st.secrets first, then fall back to os.environ.
    This allows the app to work both on Streamlit Cloud and locally.
    """
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    # Fall back to environment variables (for local development with .env)
    from dotenv import load_dotenv
    load_dotenv()
    
    value = os.getenv(key)
    if not value:
        st.error(f"Missing required credential: {key}")
        st.info("For local development: Add to .env file")
        st.info("For Streamlit Cloud: Add to Secrets in app settings")
        st.stop()
    
    return value


@st.cache_resource
def get_supabase_client() -> Client:
    """Initialize and cache Supabase client"""
    url = get_env_var("SUPABASE_URL")
    key = get_env_var("SUPABASE_ANON_KEY")
    return create_client(url, key)
