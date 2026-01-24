"""
Rainforest API Client

Handles ASIN lookups with caching to minimize API costs.
"""

import requests
import json
import sqlite3
import time
from typing import Dict, Optional
from datetime import datetime, timedelta

class ASINCache:
    """SQLite cache for ASIN lookups (30-day TTL)."""
    
    def __init__(self, db_path: str = 'data/asin_cache.db'):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database and tables."""
        import os
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS asin_lookups (
                asin TEXT,
                marketplace TEXT,
                data TEXT,
                lookup_date TIMESTAMP,
                PRIMARY KEY (asin, marketplace)
            )
        ''')
        self.conn.commit()
    
    def get(self, asin: str, marketplace: str = 'AE') -> Optional[Dict]:
        """Get cached lookup (valid for 30 days)."""
        cutoff = datetime.now() - timedelta(days=30)
        
        cursor = self.conn.execute('''
            SELECT data FROM asin_lookups
            WHERE asin = ? AND marketplace = ?
            AND lookup_date > ?
        ''', (asin.upper(), marketplace, cutoff))
        
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None
    
    def set(self, asin: str, marketplace: str, data: Dict):
        """Cache lookup result."""
        self.conn.execute('''
            INSERT OR REPLACE INTO asin_lookups
            (asin, marketplace, data, lookup_date)
            VALUES (?, ?, ?, ?)
        ''', (asin.upper(), marketplace, json.dumps(data), datetime.now()))
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_second: float = 2.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_request
        
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        self.last_request = time.time()

class RainforestClient:
    """Client for Rainforest Amazon Product API."""
    
    def __init__(self, api_key: str, cache_db: str = 'data/asin_cache.db'):
        self.api_key = api_key
        self.base_url = "https://api.rainforestapi.com/request"
        self.cache = ASINCache(cache_db)
        self.rate_limiter = RateLimiter(requests_per_second=2)
    
    def lookup_asin(self, asin: str, marketplace: str = 'AE') -> Dict:
        """
        Lookup single ASIN with caching.
        
        Args:
            asin: Amazon ASIN to lookup
            marketplace: Amazon marketplace (AE, US, UK, etc.)
            
        Returns:
            Dict with product details or error info
        """
        # Check cache first
        cached = self.cache.get(asin, marketplace)
        if cached:
            print(f"DEBUG CACHE HIT {asin}: Title='{cached.get('title', 'N/A')}', Brand='{cached.get('brand', 'N/A')}'")
            return cached
        
        # Rate limit
        self.rate_limiter.wait()
        
        # API call
        params = {
            'api_key': self.api_key,
            'type': 'product',
            'amazon_domain': f'amazon.{marketplace.lower()}',
            'asin': asin.upper()
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # AGGRESSIVE DEBUG: Print EVERYTHING
                print(f"\n{'='*80}")
                print(f"DEBUG {asin}: Full API Response")
                print(f"{'='*80}")
                print(f"Response Keys: {list(data.keys())}")
                
                if 'product' in data:
                    product = data['product']
                    print(f"Product Keys: {list(product.keys())[:20]}")  # First 20 keys
                    print(f"Title in product: {'title' in product}")
                    print(f"Brand in product: {'brand' in product}")
                    
                    # Print sample of what we see
                    if 'title' in product:
                        print(f"product['title'] = '{product['title'][:100] if product['title'] else 'EMPTY STRING'}'")
                    if 'brand' in product:
                        print(f"product['brand'] = '{product['brand']}'")
                    if 'buybox_winner' in product:
                        print(f"buybox_winner keys: {list(product['buybox_winner'].keys()) if isinstance(product['buybox_winner'], dict) else 'NOT A DICT'}")
                    print(f"{'='*80}\n")
                    
                    # Extract title with fallbacks
                    title = product.get('title', '')
                    if not title:
                        title = product.get('name', '')
                    
                    # Extract brand with multiple fallbacks (Amazon response varies)
                    brand = ''
                    if 'brand' in product and product['brand']:
                        brand = product['brand']
                    elif 'buybox_winner' in product and isinstance(product.get('buybox_winner'), dict):
                        # Sometimes brand is in buybox_winner
                        brand = product['buybox_winner'].get('brand', '')
                    
                    # If still no brand, check specifications/features
                    if not brand and 'specifications' in product:
                        specs = product.get('specifications', [])
                        if isinstance(specs, list):
                            for spec in specs:
                                if isinstance(spec, dict) and spec.get('name', '').lower() == 'brand':
                                    brand = spec.get('value', '')
                                    break
                    
                    # Last fallback: check attributes
                    if not brand and 'attributes' in product:
                        attrs = product.get('attributes', {})
                        if isinstance(attrs, dict):
                            brand = attrs.get('brand', '') or attrs.get('Brand', '')
                    
                    # Extract seller info
                    seller = ''
                    if 'buybox_winner' in product and isinstance(product.get('buybox_winner'), dict):
                        seller = product['buybox_winner'].get('name', '')
                    
                    # Extract price info
                    price = None
                    currency = None
                    if 'buybox_winner' in product and isinstance(product.get('buybox_winner'), dict):
                        buybox = product['buybox_winner']
                        if 'price' in buybox and isinstance(buybox.get('price'), dict):
                            price = buybox['price'].get('value')
                            currency = buybox['price'].get('currency')
                    
                    # DEBUG: Print extracted data
                    print(f"DEBUG {asin}: Extracted -> Title='{title[:50] if title else 'EMPTY'}', Brand='{brand or 'EMPTY'}'")
                    
                    result = {
                        'asin': asin.upper(),
                        'title': title,
                        'brand': brand,
                        'seller': seller,
                        'price': price,
                        'currency': currency,
                        'rating': product.get('rating'),
                        'reviews_count': product.get('ratings_total'),
                        'category': ' > '.join([c.get('name', '') for c in product.get('categories', [])]) if 'categories' in product else '',
                        'availability': product.get('availability', {}).get('raw', '') if isinstance(product.get('availability'), dict) else '',
                        'product_url': f"https://www.amazon.{marketplace.lower()}/dp/{asin.upper()}",
                        'status': 'success'
                    }
                    
                    # Cache result
                    self.cache.set(asin, marketplace, result)
                    return result
                else:
                    # Product not found or error in response
                    print(f"❌ {asin}: API returned no 'product' in response")
                    print(f"Response data keys: {list(data.keys())}")
                    if 'error' in data:
                        print(f"API Error: {data['error']}")
                    result = {
                        'asin': asin.upper(),
                        'status': 'not_found',
                        'error': data.get('error', 'Product not found')
                    }
                    return result
            else:
                # HTTP error
                print(f"❌ {asin}: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error response: {error_data}")
                except:
                    print(f"Error response (raw): {response.text[:200]}")
                
                result = {
                    'asin': asin.upper(),
                    'status': 'error',
                    'error': f'HTTP {response.status_code}'
                }
                return result
                
        except Exception as e:
            print(f"❌ {asin}: Exception - {str(e)}")
            import traceback
            traceback.print_exc()
            
            result = {
                'asin': asin.upper(),
                'status': 'error',
                'error': str(e)
            }
            return result
    
    def batch_lookup(self, asin_list: list, marketplace: str = 'AE') -> list:
        """
        Batch lookup multiple ASINs.
        
        Args:
            asin_list: List of ASINs to lookup
            marketplace: Amazon marketplace
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for asin in asin_list:
            result = self.lookup_asin(asin, marketplace)
            results.append(result)
        
        return results
    
    def __del__(self):
        """Cleanup cache connection."""
        if hasattr(self, 'cache'):
            self.cache.close()
