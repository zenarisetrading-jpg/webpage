
# ==========================================
# Verify Login Render (Mock)
# ==========================================
# This script imports the login function and runs it in a mocked Streamlit environment
# just to verify there are no SyntaxErrors or ImportErrors in that file path.

import sys
from pathlib import Path

# Add venv site-packages to path explicitly for this test
venv_path = Path(__file__).parent / ".venv" / "lib" / "python3.14" / "site-packages"
if venv_path.exists():
    sys.path.insert(0, str(venv_path))

# Mock Streamlit
from unittest.mock import MagicMock
import sys
sys.modules["streamlit"] = MagicMock()
import streamlit as st

# Setup mock session state
st.session_state = {}

try:
    print("🔍 Testing Login UI Import & Render Logic...")
    from ui.auth.login import render_login
    
    # Try calling it (it will use mock st commands)
    render_login()
    
    print("✅ render_login() executed without runtime errors.")
except Exception as e:
    print(f"❌ Login Render Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
