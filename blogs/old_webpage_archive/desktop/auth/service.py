"""
Authentication service handling all Supabase auth operations.
"""
import streamlit as st
from typing import Optional
from auth.config import get_supabase_client


class AuthService:
    """Handles all authentication operations with Supabase"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    def sign_up(self, email: str, password: str) -> dict:
        """
        Register a new user with email and password.
        
        Args:
            email: User's email address
            password: Password (minimum 8 characters)
            
        Returns:
            dict with 'success' boolean and either 'user' or 'error'
        """
        # Validate password length
        if len(password) < 8:
            return {"success": False, "error": "Password must be at least 8 characters"}
        
        try:
            response = self.client.auth.sign_up({
                "email": email.strip().lower(),
                "password": password
            })
            
            if response.user:
                return {
                    "success": True, 
                    "user": response.user,
                    "message": "Account created! Please check your email to verify your account."
                }
            else:
                return {"success": False, "error": "Registration failed. Please try again."}
                
        except Exception as e:
            error_msg = str(e).lower()
            if "already registered" in error_msg or "already exists" in error_msg:
                return {"success": False, "error": "An account with this email already exists"}
            elif "invalid email" in error_msg:
                return {"success": False, "error": "Please enter a valid email address"}
            elif "weak password" in error_msg:
                return {"success": False, "error": "Password is too weak. Use at least 8 characters with mixed case and numbers."}
            else:
                return {"success": False, "error": f"Registration failed: {str(e)}"}
    
    def sign_in(self, email: str, password: str) -> dict:
        """
        Authenticate an existing user.
        
        Args:
            email: User's email address
            password: User's password
            
        Returns:
            dict with 'success' boolean and either 'user' or 'error'
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email.strip().lower(),
                "password": password
            })
            
            if response.user and response.session:
                # Store in session state
                st.session_state["user"] = response.user
                st.session_state["session"] = response.session
                st.session_state["access_token"] = response.session.access_token
                
                return {"success": True, "user": response.user}
            else:
                return {"success": False, "error": "Login failed. Please try again."}
                
        except Exception as e:
            error_msg = str(e).lower()
            if "invalid" in error_msg or "credentials" in error_msg:
                return {"success": False, "error": "Invalid email or password"}
            elif "not confirmed" in error_msg:
                return {"success": False, "error": "Please verify your email before logging in"}
            else:
                return {"success": False, "error": "Login failed. Please check your credentials."}
    
    def sign_out(self) -> None:
        """Sign out the current user and clear session state"""
        try:
            self.client.auth.sign_out()
        except Exception:
            pass  # Ignore errors during sign out
        
        # Clear all auth-related session state
        keys_to_clear = ["user", "session", "access_token", "auth_redirect"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    def get_current_user(self) -> Optional[dict]:
        """Get the currently authenticated user from session state"""
        return st.session_state.get("user")
    
    def is_authenticated(self) -> bool:
        """Check if a user is currently authenticated"""
        user = st.session_state.get("user")
        return user is not None
    
    def reset_password(self, email: str) -> dict:
        """
        Send a password reset email to the user.
        
        Args:
            email: User's email address
            
        Returns:
            dict with 'success' boolean and optional 'error'
        """
        try:
            self.client.auth.reset_password_email(email.strip().lower())
            return {
                "success": True, 
                "message": "Password reset link sent! Check your email."
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to send reset email: {str(e)}"}
    
    def update_password(self, new_password: str) -> dict:
        """
        Update password for the currently authenticated user.
        
        Args:
            new_password: New password (minimum 8 characters)
            
        Returns:
            dict with 'success' boolean and optional 'error'
        """
        if len(new_password) < 8:
            return {"success": False, "error": "Password must be at least 8 characters"}
        
        try:
            self.client.auth.update_user({"password": new_password})
            return {"success": True, "message": "Password updated successfully!"}
        except Exception as e:
            return {"success": False, "error": f"Failed to update password: {str(e)}"}
    
    def get_user_email(self) -> Optional[str]:
        """Get the email of the currently authenticated user"""
        user = self.get_current_user()
        if user:
            return user.email
        return None
    
    def update_user_metadata(self, metadata: dict) -> dict:
        """
        Update user metadata (profile information) for the current user.
        
        Args:
            metadata: Dictionary with profile fields (full_name, phone, company, role, etc.)
            
        Returns:
            dict with 'success' boolean and optional 'error'
        """
        try:
            response = self.client.auth.update_user({
                "data": metadata
            })
            
            if response.user:
                # Update the user in session state with new metadata
                st.session_state["user"] = response.user
                return {"success": True, "message": "Profile updated successfully!"}
            else:
                return {"success": False, "error": "Failed to update profile"}
                
        except Exception as e:
            return {"success": False, "error": f"Failed to update profile: {str(e)}"}
