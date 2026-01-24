import os
import psycopg2
import sys

def get_db_url():
    # Try reading .env file
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('DATABASE_URL='):
                    return line.strip().split('=', 1)[1].strip('\"\'')
    return os.environ.get('DATABASE_URL')

def main():
    print("Checking SKU Map for s2c_uae_test...")
    
    db_url = get_db_url()
    if not db_url:
        print("Error: Could not find DATABASE_URL")
        return

    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        client_id = 's2c_uae_test'
        
        # Check count
        query = "SELECT COUNT(*) FROM advertised_product_cache WHERE client_id = %s"
        cursor.execute(query, (client_id,))
        count = cursor.fetchone()[0]
        print(f"Row count in advertised_product_cache: {count}")
        
        if count > 0:
            # Show sample
            cursor.execute("SELECT campaign_name, ad_group_name, sku, asin FROM advertised_product_cache WHERE client_id = %s LIMIT 3", (client_id,))
            rows = cursor.fetchall()
            print("\nSample Data:")
            for row in rows:
                print(row)
        else:
            print("❌ No SKU data found for this client.")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
