"""
Diagnostic script to compare SKUs between Category Map and Advertised Product Report.
Run this to see why category matching is failing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.db_manager import get_db_manager

def compare_skus():
    db = get_db_manager(test_mode=False)
    
    # First, discover what client_ids exist
    print("=" * 60)
    print("DISCOVERING CLIENT IDS IN DATABASE")
    print("=" * 60)
    
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT client_id FROM category_mappings")
            cat_clients = [r[0] if isinstance(r, tuple) else r['client_id'] for r in cursor.fetchall()]
            print(f"Category Mappings - Client IDs: {cat_clients}")
            
            cursor.execute("SELECT DISTINCT client_id FROM advertised_product_cache")
            adv_clients = [r[0] if isinstance(r, tuple) else r['client_id'] for r in cursor.fetchall()]
            print(f"Advertised Product Cache - Client IDs: {adv_clients}")
    except Exception as e:
        print(f"Error discovering clients: {e}")
        cat_clients = []
        adv_clients = []
    
    # Use first available or default
    client_id = cat_clients[0] if cat_clients else (adv_clients[0] if adv_clients else 'default_client')
    print(f"\nUsing client_id: {client_id}")
    
    print("\n" + "=" * 60)
    print("SKU COMPARISON: Category Map vs Advertised Product")
    print("=" * 60)
    
    # Get category mappings
    cat_map = db.get_category_mappings(client_id)
    if cat_map is None or cat_map.empty:
        print("\n❌ No category mappings found in database!")
        cat_skus = set()
    else:
        print(f"\n📁 CATEGORY MAP: {len(cat_map)} rows")
        # Find SKU column
        sku_col = next((c for c in cat_map.columns if 'sku' in c.lower()), None)
        if sku_col:
            cat_skus = set(cat_map[sku_col].astype(str).str.strip().str.lower().unique())
            print(f"   SKU column: '{sku_col}'")
            print(f"   Unique SKUs: {len(cat_skus)}")
            print(f"\n   Sample SKUs (first 20):")
            for sku in sorted(list(cat_skus))[:20]:
                print(f"      - {sku}")
        else:
            print("   ❌ No SKU column found!")
            cat_skus = set()
    
    # Get advertised product map (correct method name)
    adv_map = db.get_advertised_product_map(client_id)
    if adv_map is None or adv_map.empty:
        print("\n❌ No advertised product data found in database!")
        adv_skus = set()
    else:
        print(f"\n📦 ADVERTISED PRODUCT: {len(adv_map)} rows")
        # Find SKU column
        sku_col = next((c for c in adv_map.columns if 'sku' in c.lower()), None)
        if sku_col:
            adv_skus = set(adv_map[sku_col].astype(str).str.strip().str.lower().unique())
            print(f"   SKU column: '{sku_col}'")
            print(f"   Unique SKUs: {len(adv_skus)}")
            print(f"\n   Sample SKUs (first 20):")
            for sku in sorted(list(adv_skus))[:20]:
                print(f"      - {sku}")
        else:
            print("   ❌ No SKU column found!")
            adv_skus = set()
    
    print("\n" + "=" * 60)
    print("MATCH ANALYSIS")
    print("=" * 60)
    
    if cat_skus and adv_skus:
        matches = cat_skus & adv_skus
        only_in_cat = cat_skus - adv_skus
        only_in_adv = adv_skus - cat_skus
        
        print(f"\n✅ MATCHING SKUs: {len(matches)}")
        if matches:
            for sku in sorted(list(matches))[:10]:
                print(f"   - {sku}")
            if len(matches) > 10:
                print(f"   ... and {len(matches) - 10} more")
        
        print(f"\n⚠️  In Category Map ONLY (not in Advertised): {len(only_in_cat)}")
        for sku in sorted(list(only_in_cat))[:10]:
            print(f"   - {sku}")
        if len(only_in_cat) > 10:
            print(f"   ... and {len(only_in_cat) - 10} more")
        
        print(f"\n⚠️  In Advertised ONLY (not in Category Map): {len(only_in_adv)}")
        for sku in sorted(list(only_in_adv))[:10]:
            print(f"   - {sku}")
        if len(only_in_adv) > 10:
            print(f"   ... and {len(only_in_adv) - 10} more")
        
        # Match rate
        total_unique = len(cat_skus | adv_skus)
        match_rate = len(matches) / len(adv_skus) * 100 if adv_skus else 0
        
        print(f"\n📊 Match Rate: {match_rate:.1f}% ({len(matches)}/{len(adv_skus)} advertised SKUs found in category map)")
    else:
        print("\n❌ Cannot compare - one or both sources are empty!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    compare_skus()
