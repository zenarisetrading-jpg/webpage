
import pandas as pd
import re
import difflib
from collections import defaultdict

class ExactMatcher:
    """Fuzzy matcher for detecting existing exact match keywords."""
    
    def __init__(self, df: pd.DataFrame):
        match_col = "Match Type" if "Match Type" in df.columns else "Match"
        if match_col not in df.columns:
            self.exact_keywords = set()
            return
        match_types = df[match_col].astype(str).fillna("")
        exact_rows = df[match_types.str.contains("exact", case=False, na=False)]
        term_col = "Customer Search Term" if "Customer Search Term" in df.columns else "Term"
        self.exact_keywords = set(exact_rows[term_col].astype(str).apply(self.normalize_text).unique())
        self.token_index = defaultdict(set)
        for kw in self.exact_keywords:
            tokens = self.get_tokens(kw)
            for t in tokens:
                self.token_index[t].add(kw)

    def normalize_text(self, s: str) -> str:
        if not isinstance(s, str): return ""
        return re.sub(r'[^a-zA-Z0-9\s]', '', s.lower())

    def get_tokens(self, s: str) -> set:
        return set(self.normalize_text(s).split())

    def find_match(self, term: str, threshold: float = 0.90) -> tuple[str | None, float]:
        norm_term = self.normalize_text(str(term))
        if not norm_term: return None, 0.0
        if norm_term in self.exact_keywords: return norm_term, 1.0
        
        term_tokens = self.get_tokens(norm_term)
        candidates = set()
        for t in term_tokens:
            if t in self.token_index:
                candidates.update(self.token_index[t])
        
        if not candidates: return None, 0.0
        
        best_match = None
        best_score = 0.0
        for cand in candidates:
            score = difflib.SequenceMatcher(None, norm_term, cand).ratio()
            if score > best_score:
                best_score = score
                best_match = cand
        
        if best_score >= threshold: return best_match, best_score
        return None, 0.0
