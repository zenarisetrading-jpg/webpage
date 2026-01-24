"""
Anthropic Claude API Client

Handles AI-powered cluster analysis and strategic recommendations.
"""

import requests
import json
import pandas as pd
from typing import Dict, List, Any

class AnthropicClient:
    """Client for Anthropic Claude API."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def analyze_clusters(self, cluster_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze search term clusters and return strategic insights.
        
        Args:
            cluster_df: DataFrame with cluster summaries
            
        Returns:
            Dict with opportunities and mismatches
        """
        # Prepare cluster data for Claude
        cluster_summaries = []
        
        for _, row in cluster_df.iterrows():
            cluster_summaries.append({
                'cluster_id': int(row['cluster_id']),
                'top_terms': row['top_terms'][:5],
                'impressions': int(row['impressions']),
                'clicks': int(row['clicks']),
                'spend': float(row['spend']),
                'orders': int(row['orders']),
                'wasted_spend': float(row['wasted_spend']),
                'ctr': float(row['ctr']),
                'cvr': float(row['cvr'])
            })
        
        # Build prompt
        prompt = self._build_analysis_prompt(cluster_summaries)
        
        # Call Claude API
        response = self._call_api(prompt)
        
        # Parse response
        insights = self._parse_response(response)
        
        return insights
    
    def _build_analysis_prompt(self, clusters: List[Dict]) -> str:
        """Build analysis prompt for Claude."""
        
        clusters_json = json.dumps(clusters, indent=2)
        
        prompt = f"""You are an expert Amazon PPC strategist analyzing search term clusters.

I have clustered customer search terms into {len(clusters)} semantic groups. Your job is to:
1. Identify the TOP 2 OPPORTUNITIES (highest revenue potential)
2. Identify the TOP 2 INTENT MISMATCHES (highest wasted spend)

For each, provide:
- Title (concise, descriptive)
- Description (what customers want, 2-3 sentences)
- Action (specific fix, 1 sentence)
- Potential/Wasted (revenue estimate in AED)

Here are the clusters:

{clusters_json}

Respond in this EXACT JSON format:
{{
  "opportunities": [
    {{
      "cluster_id": <id>,
      "title": "...",
      "description": "...",
      "action": "...",
      "potential": "AED X,XXX/month"
    }},
    {{...}}
  ],
  "mismatches": [
    {{
      "cluster_id": <id>,
      "title": "...",
      "description": "...",
      "fix": "...",
      "wasted": "AED X,XXX/month"
    }},
    {{...}}
  ]
}}

Focus on:
- Actionable insights (not generic advice)
- Quantified impact (use the actual numbers)
- Specific recommendations (not vague)
"""
        return prompt
    
    def _call_api(self, prompt: str) -> Dict:
        """Call Claude API."""
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")
        
        except Exception as e:
            raise Exception(f"Failed to call Claude API: {str(e)}")
    
    def _parse_response(self, response: Dict) -> Dict[str, Any]:
        """Parse Claude API response."""
        
        try:
            # Extract content
            content = response['content'][0]['text']
            
            # Claude sometimes wraps JSON in markdown code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            insights = json.loads(content)
            
            return insights
        
        except Exception as e:
            # Fallback if parsing fails
            return {
                'opportunities': [
                    {
                        'cluster_id': 0,
                        'title': 'Analysis Error',
                        'description': f'Failed to parse Claude response: {str(e)}',
                        'action': 'Check API response format',
                        'potential': 'Unknown'
                    }
                ],
                'mismatches': [
                    {
                        'cluster_id': 0,
                        'title': 'Analysis Error',
                        'description': 'Failed to parse response',
                        'fix': 'Review API configuration',
                        'wasted': 'Unknown'
                    }
                ]
            }
