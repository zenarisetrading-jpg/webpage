"""
PPC Audit - All-in-One Server
Runs both backend API and frontend server in one command

Usage: python run_audit.py
Then open: http://localhost:8000
"""

from flask import Flask, request, jsonify, make_response, send_from_directory
import pandas as pd
import re
from io import BytesIO
import os
from threading import Thread

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=SCRIPT_DIR)

# Manual CORS handling
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Config
CONFIG = {
    'HARVEST_MIN_CLICKS': 10,
    'HARVEST_MIN_ORDERS': 3,
    'HARVEST_MIN_SALES': 150.00,
    'NEGATIVE_CLICKS_THRESHOLD': 10,
    'NEGATIVE_SPEND_THRESHOLD': 10.00,
    'ROAS_TARGET': 2.50
}

def parse_search_term_report(file_bytes, filename):
    """Parse Excel or CSV Search Term Report"""
    
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        df = pd.read_csv(BytesIO(file_bytes))
    
    df.columns = df.columns.str.strip()
    
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'customer search term' in col_lower or col_lower == 'search term':
            col_map['search_term'] = col
        elif 'campaign name' in col_lower:
            col_map['campaign'] = col
        elif 'ad group name' in col_lower:
            col_map['ad_group'] = col
        elif col_lower == 'clicks':
            col_map['clicks'] = col
        elif col_lower == 'spend':
            col_map['spend'] = col
        elif '7 day total sales' in col_lower or col_lower == 'sales':
            col_map['sales'] = col
        elif '7 day total orders' in col_lower or 'orders' in col_lower:
            col_map['orders'] = col
        elif col_lower == 'impressions':
            col_map['impressions'] = col
        elif 'match type' in col_lower:
            col_map['match_type'] = col
    
    required = ['search_term', 'clicks', 'spend']
    missing = [r for r in required if r not in col_map]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    rename_dict = {col_map[k]: k for k in col_map}
    df = df.rename(columns=rename_dict)
    
    for col, default in [('sales', 0), ('orders', 0), ('impressions', 0)]:
        if col not in df.columns:
            df[col] = default
    
    if 'campaign' not in df.columns:
        df['campaign'] = 'Unknown'
    if 'ad_group' not in df.columns:
        df['ad_group'] = 'Unknown'
    if 'match_type' not in df.columns:
        df['match_type'] = ''
    
    numeric_cols = ['clicks', 'spend', 'sales', 'orders', 'impressions']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df[(df['clicks'] > 0) | (df['spend'] > 0) | (df['impressions'] > 0)].copy()
    
    agg_cols = ['campaign', 'ad_group', 'search_term', 'match_type']
    df = df.groupby(agg_cols, as_index=False).agg({
        'clicks': 'sum',
        'spend': 'sum',
        'sales': 'sum',
        'orders': 'sum',
        'impressions': 'sum'
    })
    
    return df

def analyze_data(df):
    """Compute audit results"""
    
    totals = {
        'spend': float(df['spend'].sum()),
        'sales': float(df['sales'].sum()),
        'clicks': int(df['clicks'].sum()),
        'orders': int(df['orders'].sum()),
        'impressions': int(df['impressions'].sum())
    }
    
    totals['roas'] = totals['sales'] / totals['spend'] if totals['spend'] > 0 else 0
    totals['acos'] = (totals['spend'] / totals['sales'] * 100) if totals['sales'] > 0 else 0
    totals['ctr'] = (totals['clicks'] / totals['impressions'] * 100) if totals['impressions'] > 0 else 0
    
    asin_pattern = r'B0[A-Z0-9]{8}'
    df['has_asin'] = df['search_term'].str.contains(asin_pattern, case=False, na=False, regex=True)
    competitor_asins = df[df['has_asin'] & (df['spend'] > 0)].copy()
    competitor_waste = float(competitor_asins['spend'].sum())
    
    # Count unique ASINs safely
    if len(competitor_asins) > 0:
        try:
            # Use named group for extraction
            competitor_count = len(competitor_asins['search_term'].str.findall(asin_pattern).explode().unique())
        except:
            competitor_count = len(competitor_asins)
    else:
        competitor_count = 0
    
    zero_conv = df[
        (df['sales'] == 0) & 
        ((df['clicks'] >= CONFIG['NEGATIVE_CLICKS_THRESHOLD']) | 
         (df['spend'] >= CONFIG['NEGATIVE_SPEND_THRESHOLD']))
    ].copy()
    zero_conv_waste = float(zero_conv['spend'].sum())
    
    df['roas'] = df.apply(lambda r: r['sales'] / r['spend'] if r['spend'] > 0 else 0, axis=1)
    df['is_exact'] = df['campaign'].str.lower().str.contains('exact', na=False)
    
    harvest = df[
        (df['clicks'] >= CONFIG['HARVEST_MIN_CLICKS']) &
        (df['orders'] >= CONFIG['HARVEST_MIN_ORDERS']) &
        (df['sales'] >= CONFIG['HARVEST_MIN_SALES']) &
        (df['roas'] > 0) &
        (~df['is_exact'])
    ].copy()
    harvest_gain = float(harvest['sales'].sum() * 0.20)
    
    target_acos = (1 / CONFIG['ROAS_TARGET']) * 100
    bid_waste = totals['spend'] * ((totals['acos'] - target_acos) / 100) * 0.5 if totals['acos'] > target_acos else 0
    
    df['ctr'] = df.apply(lambda r: (r['clicks'] / r['impressions'] * 100) if r['impressions'] > 0 else 0, axis=1)
    neg_gaps = df[
        (df['sales'] == 0) &
        (df['ctr'] < 0.5) &
        (df['spend'] > 5) &
        (~df.index.isin(zero_conv.index))
    ].copy()
    neg_gap_waste = float(neg_gaps['spend'].sum())
    
    waste_pct = ((competitor_waste + zero_conv_waste + bid_waste + neg_gap_waste) / totals['spend'] * 100) if totals['spend'] > 0 else 0
    health_score = max(45, min(100, 100 - waste_pct))
    
    component_scores = {
        'wastedSpend': max(0, 100 - (waste_pct * 1.5)),
        'harvestEfficiency': max(40, 100 - (len(harvest) * 2)) if len(harvest) > 0 else 85,
        'negativeHygiene': max(40, 100 - (len(zero_conv) * 0.5)) if len(zero_conv) > 0 else 90,
        'bidOptimization': max(40, 100 - abs(totals['roas'] - CONFIG['ROAS_TARGET']) * 10),
        'campaignStructure': 75
    }
    
    issues = []
    
    if competitor_waste > 0:
        top_items = competitor_asins.nlargest(5, 'spend')[['search_term', 'spend']].to_dict('records')
        issues.append({
            'priority': 'high',
            'title': 'Competitor ASIN Bleed',
            'amount': round(competitor_waste, 2),
            'count': int(competitor_count),
            'type': 'waste',
            'description': f'{competitor_count} competitor ASINs found',
            'items': [{'term': item['search_term'], 'spend': round(item['spend'], 2)} for item in top_items]
        })
    
    if zero_conv_waste > 0:
        top_items = zero_conv.nlargest(5, 'spend')[['search_term', 'spend', 'clicks']].to_dict('records')
        issues.append({
            'priority': 'high',
            'title': 'Zero-Conversion Keywords',
            'amount': round(zero_conv_waste, 2),
            'count': len(zero_conv),
            'type': 'waste',
            'description': f'{len(zero_conv)} keywords with ≥{CONFIG["NEGATIVE_CLICKS_THRESHOLD"]} clicks or ≥${CONFIG["NEGATIVE_SPEND_THRESHOLD"]} spend, 0 sales',
            'items': [{'term': item['search_term'], 'spend': round(item['spend'], 2), 'clicks': int(item['clicks'])} for item in top_items]
        })
    
    if harvest_gain > 0:
        top_items = harvest.nlargest(5, 'sales')[['search_term', 'sales', 'orders', 'roas']].to_dict('records')
        issues.append({
            'priority': 'high',
            'title': 'Missed Harvest Opportunities',
            'amount': round(harvest_gain, 2),
            'count': len(harvest),
            'type': 'gain',
            'description': f'{len(harvest)} high-performing terms',
            'items': [{'term': item['search_term'], 'sales': round(item['sales'], 2), 'orders': int(item['orders']), 'roas': round(item['roas'], 2)} for item in top_items]
        })
    
    if bid_waste > 0:
        issues.append({
            'priority': 'medium',
            'title': 'Bid Inefficiency',
            'amount': round(bid_waste, 2),
            'count': None,
            'type': 'waste',
            'description': f'Current ACOS {totals["acos"]:.1f}% vs Target {target_acos:.1f}%',
            'items': []
        })
    
    if neg_gap_waste > 0:
        top_items = neg_gaps.nlargest(5, 'spend')[['search_term', 'spend', 'ctr']].to_dict('records')
        issues.append({
            'priority': 'medium',
            'title': 'Additional Negative Gaps',
            'amount': round(neg_gap_waste, 2),
            'count': len(neg_gaps),
            'type': 'waste',
            'description': f'{len(neg_gaps)} low-CTR terms',
            'items': [{'term': item['search_term'], 'spend': round(item['spend'], 2), 'ctr': round(item['ctr'], 2)} for item in top_items]
        })
    
    return {
        'healthScore': round(health_score),
        'componentScores': {k: round(v) for k, v in component_scores.items()},
        'totals': {k: round(v, 2) if isinstance(v, float) else v for k, v in totals.items()},
        'issues': issues,
        'totalOpportunity': round(competitor_waste + zero_conv_waste + harvest_gain + bid_waste + neg_gap_waste, 2),
        'dataQuality': {
            'totalRows': len(df),
            'validRows': len(df[df['clicks'] > 0]),
            'dateRange': 'Aggregated from all dates'
        }
    }

# Routes
@app.route('/')
def index():
    return send_from_directory(SCRIPT_DIR, 'index.html')

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return make_response('', 204)
    return jsonify({'status': 'ok'})

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return make_response('', 204)
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_bytes = file.read()
        df = parse_search_term_report(file_bytes, file.filename)
        results = analyze_data(df)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("=" * 60)
    print("PPC Audit - All-in-One Server")
    print("=" * 60)
    print("✓ Backend API running on port 8080")
    print("✓ Frontend running on port 8080")
    print("\nOpen: http://localhost:8080")
    print("=" * 60)
    
    app.run(debug=True, port=8080, host='0.0.0.0')