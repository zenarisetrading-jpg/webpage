"""
Base Feature Template

All features inherit from this class to maintain consistency.
This provides:
- Standard initialization
- Config management
- Data validation
- Output generation patterns
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseFeature(ABC):
    """Base class for all PPC Suite features."""
    
    def __init__(self):
        """Initialize feature with config and data."""
        self.config = self.load_config()
        self.data = None
        self.results = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load feature-specific configuration."""
        # Override in subclass if needed
        return {}
    
    @abstractmethod
    def render_ui(self):
        """Render the feature's user interface."""
        pass

    def render_header(self, title: str, icon_name: str = None):
        """Render standard header with optional icon using HTML/CSS for alignment."""
        if icon_name:
            import base64
            import os
            
            try:
                with open(f"assets/icons/{icon_name}.png", "rb") as f:
                    data = f.read()
                    encoded = base64.b64encode(data).decode()
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{encoded}" width="50" style="margin-right: 15px;">
                    <h1 style="margin: 0; padding: 0; line-height: 1.2;">{title}</h1>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                st.title(f"⚠️ {title}") # Fallback
        else:
            st.title(title)
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate input data.
        
        Returns:
            (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Core business logic - analyze the data.
        
        Returns:
            Dictionary of results
        """
        pass
    
    def generate_output(self, results: Dict[str, Any]) -> bytes:
        """
        Generate downloadable output file.
        
        Returns:
            Excel file as bytes
        """
        # Default implementation - override if needed
        from utils.formatters import dataframe_to_excel
        return dataframe_to_excel(results.get('data'))
    
    def run(self):
        """Main execution flow - orchestrates everything."""
        try:
            # Render UI
            self.render_ui()
            
            # Check if user uploaded data
            if self.data is None:
                return
            
            # Validate data
            is_valid, error_msg = self.validate_data(self.data)
            if not is_valid:
                st.error(f"❌ {error_msg}")
                return
            
            # Analyze
            with st.spinner("Analyzing..."):
                self.results = self.analyze(self.data)
            
            # Show results
            self.display_results(self.results)
            
            # Offer download
            if self.results:
                output = self.generate_output(self.results)
                st.download_button(
                    label="📥 Download Results",
                    data=output,
                    file_name=f"{self.__class__.__name__}_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    def display_results(self, results: Dict[str, Any]):
        """Display results in UI - override in subclass."""
        st.success("✅ Analysis complete!")
        st.json(results)
