
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import re
from core.data_loader import safe_numeric, SmartMapper

file_path = "Sponsored_Products_Search_term_report (4).xlsx"
print(f"Loading {file_path}...")

try:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    print("Columns found:", list(df.columns))
    
    # Identify Spend column via SmartMapper
    col_map = SmartMapper.map_columns(df)
    spend_col = col_map.get("Spend")
    print(f"Mapped 'Spend' to: '{spend_col}'")
    
    if spend_col:
        print("\n--- Raw Sample (Head) ---")
        print(df[spend_col].head(5))
        
        print("\n--- Raw Sample (Tail) ---")
        print(df[spend_col].tail(5))
        
        # Test safe_numeric
        print("\n--- Applying safe_numeric ---")
        cleaned = safe_numeric(df[spend_col])
        print(cleaned.head(5))
        print(f"Total Spend: {cleaned.sum()}")
        
        # Debug regex directly
        print("\n--- Regex Debug ---")
        sample = df[spend_col].iloc[0]
        print(f"Sample Value: '{sample}' (Type: {type(sample)})")
        if isinstance(sample, str):
            replaced = re.sub(r'[^0-9.-]', '', sample)
            print(f"After Regex: '{replaced}'")
            try:
                val = float(replaced)
                print(f"Converted Float: {val}")
            except Exception as e:
                print(f"Conversion Failed: {e}")
                
    else:
        print("❌ Could not map 'Spend' column.")
        
except Exception as e:
    print(f"Error: {e}")
