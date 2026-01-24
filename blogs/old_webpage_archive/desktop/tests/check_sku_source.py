"""
Check where SKU_advertised values like 40OZDARKRAINBOW are coming from.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.db_manager import get_db_manager

def check_sku_source():
    db = get_db_manager(test_mode=False)
    client_id = 'demo_account_2'
    
    print("=" * 60)
    print("CHECKING SKU SOURCES FOR: 40OZDARKRAINBOW")
    print("=" * 60)
    
    # Check advertised product cache
    print("\n1. ADVERTISED PRODUCT CACHE:")
    adv = db.get_advertised_product_map(client_id)
    if adv is not None and not adv.empty:
        sku_col = 'SKU' if 'SKU' in adv.columns else 'sku'
        if sku_col in adv.columns:
            matches = adv[adv[sku_col].astype(str).str.upper().str.contains('40OZ', na=False)]
            print(f"   Total rows: {len(adv)}")
            print(f"   Rows containing '40OZ': {len(matches)}")
            if len(matches) > 0:
                print(f"   Sample matches: {matches[sku_col].head(5).tolist()}")
    else:
        print("   No data found")
    
    # Check bulk mappings
    print("\n2. BULK MAPPINGS:")
    bulk = db.get_bulk_mapping(client_id)
    if bulk is not None and not bulk.empty:
        print(f"   Columns: {list(bulk.columns)}")
        sku_cols = [c for c in bulk.columns if 'sku' in c.lower()]
        if sku_cols:
            for sku_col in sku_cols:
                matches = bulk[bulk[sku_col].astype(str).str.upper().str.contains('40OZ', na=False)]
                print(f"   SKU column '{sku_col}': {len(matches)} rows containing '40OZ'")
                if len(matches) > 0:
                    print(f"   Sample: {matches[sku_col].head(5).tolist()}")
        else:
            print("   No SKU columns found")
    else:
        print("   No data found")
    
    # Check category mappings
    print("\n3. CATEGORY MAPPINGS:")
    cat = db.get_category_mappings(client_id)
    if cat is not None and not cat.empty:
        sku_col = 'SKU' if 'SKU' in cat.columns else 'sku'
        if sku_col in cat.columns:
            matches = cat[cat[sku_col].astype(str).str.upper().str.contains('40OZ', na=False)]
            print(f"   Total rows: {len(cat)}")
            print(f"   Rows containing '40OZ': {len(matches)}")
            if len(matches) > 0:
                print(f"   Sample matches: {matches[sku_col].head(5).tolist()}")
            else:
                print("   ⚠️ NO 40OZ SKUs in category map!")
    else:
        print("   No data found")

if __name__ == "__main__":
    check_sku_source()
