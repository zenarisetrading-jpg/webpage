# Amazon Bulk Upload Validation - Quick Reference Cheatsheet

## 🏗️ ARCHITECTURE: VALIDATE AT SOURCE

**Don't validate at export. Validate when recommendations are generated.**

```
Optimizer generates recommendation
         ↓
   VALIDATE IMMEDIATELY
         ↓
    ┌────┴────┐
    ↓         ↓
  ✅ Valid   ❌ Invalid
    ↓         ↓
 Can export   BLOCKED
              (shown greyed out)
```

**Key Classes (from bulk_validation_spec.py):**
```python
from bulk_validation_spec import (
    OptimizationRecommendation,  # Wrapper for each recommendation
    RecommendationType,          # NEGATIVE_ISOLATION, NEGATIVE_BLEEDER, etc.
    validate_recommendation,     # Call immediately after creating rec
    recommendations_to_bulk_sheet,  # Convert valid recs to export rows
)
```

---

## 🚨 CRITICAL RULES (Will Cause Upload Failure)

### Never Modify These Columns
- `Record ID` - Internal Amazon ID, changing breaks everything
- `Record Type` - Cannot change entity type
- `Campaign Targeting Type` - Auto/Manual locked at creation
- `Bidding Strategy` - Locked at creation

### Required for ALL Record Types
| Column | Notes |
|--------|-------|
| Record Type | Must match exactly: "Campaign", "Ad Group", "Keyword", "Ad", etc. |
| Campaign | Case-sensitive! Must match parent exactly |
| Status | enabled/paused/archived (lowercase accepted) |

---

## 📋 RECORD TYPE: Campaign

### Required Columns
| Column | Accepted Values | Limits |
|--------|-----------------|--------|
| Record Type | "Campaign" | - |
| Campaign | Text | Max 128 chars |
| Campaign Daily Budget | Number | 1 - 1,000,000 (USD) |
| Campaign Start Date | mm/dd/yyyy | Cannot be past |
| Campaign Targeting Type | "Auto" or "Manual" | Immutable |
| Campaign Status | enabled/paused/archived | - |
| Bidding Strategy | See below | Immutable |

### Bidding Strategy Values (exact)
- `Fixed Bids`
- `Dynamic bidding (up and down)`
- `Dynamic bidding (down only)`

### Optional
- Portfolio ID
- Campaign End Date (mm/dd/yyyy, must be after start)

---

## 📋 RECORD TYPE: Ad Group

### Required Columns
| Column | Accepted Values | Limits |
|--------|-----------------|--------|
| Record Type | "Ad Group" | - |
| Campaign | Must match parent | Case-sensitive |
| Ad Group | Text | Max 255 chars |
| Max Bid | Decimal | 0.02 - 1,000 (USD) |
| Ad Group Status | enabled/paused/archived | - |

### For Updates Only
- Record ID (required)
- Campaign ID (required)

---

## 📋 RECORD TYPE: Keyword

### Required Columns
| Column | Accepted Values | Limits |
|--------|-----------------|--------|
| Record Type | "Keyword" | - |
| Campaign | Must match parent | Case-sensitive |
| Ad Group | Must match parent* | *Not for campaign negatives |
| Keyword or Product Targeting | Text | Max 80 chars |
| Match Type | See below | - |
| Status | enabled/paused/archived | - |

### Match Type Values
**Biddable (require Max Bid):**
- `broad`
- `phrase`
- `exact`

**Negative (Ad Group level):**
- `negative phrase`
- `negative exact`

**Negative (Campaign level):**
- `campaign negative phrase`
- `campaign negative exact`

### Keyword Word Limits
- Positive keywords: Max 10 words
- Negative keywords: Max 4 words
- Negative exact: Max 10 words

### Invalid Characters
❌ No special chars except hyphens: `#`, `@`, `*`, `$`, etc.

---

## 📋 RECORD TYPE: Product Targeting

### Keyword or Product Targeting Formats
```
asin="B0XXXXXXXXX"
category="Category Name"
brand="Brand Name"
```

### Product Targeting ID (Auto campaigns only)
- `close-match`
- `substitutes`
- `loose-match`
- `complements`

### Match Type Values
- `Targeting Expression`
- `negativetargetingexpression`
- `Targeting Expression Predefined`

---

## 📋 RECORD TYPE: Campaign By Placement

### Placement Type (exact values!)
✅ `Top of search (page 1)`
✅ `Product Pages`

❌ NOT: "Top of Search", "Product Detail Page", "Rest of search"

### Increase Bids By Placement
- Range: 0 - 900
- No decimals
- % sign optional

---

## 📋 RECORD TYPE: Ad

### Required Columns
| Column | Notes |
|--------|-------|
| Record Type | "Ad" |
| Campaign | Case-sensitive match |
| Ad Group | Case-sensitive match |
| SKU | For sellers (vendors use ASIN) |
| Status | enabled/paused/archived |

---

## 💰 CURRENCY-SPECIFIC LIMITS

### Sponsored Products

| Currency | Min Budget | Max Budget | Min Bid | Max Bid |
|----------|------------|------------|---------|---------|
| USD | 1 | 1,000,000 | 0.02 | 1,000 |
| AED | 4 | 3,700,000 | 0.24 | 3,670 |
| GBP | 1 | 1,000,000 | 0.02 | 1,000 |
| EUR | 1 | 1,000,000 | 0.02 | 1,000 |
| INR | 500 | 21,000,000 | 1 | 5,000 |
| JPY | 100 | 21,000,000 | 2 | 100,000 |
| AUD | 1.4 | 1,500,000 | 0.1 | 1,410 |

---

## ⚠️ COMMON ERROR CODES

### Fatal (File Rejected)
| Code | Issue |
|------|-------|
| 1021 | Invalid value submitted |
| 1025 | Invalid placement type |
| 2001 | Campaign creation failed |
| 2002 | Missing required columns |

### Amazon Error Message Patterns
- `cannot be created, missing columns` → Check required fields
- `duplicate row` → Same entity twice in file
- `Invalid value submitted for X` → Check allowed values exactly
- `PORTFOLIO_BUDGET_POLICY_INVALID` → Delete Portfolio tab rows

---

## 🔄 VALIDATION WORKFLOW

```
1. FILE LEVEL
   ├── Check format (.xlsx, .xls, .csv)
   ├── Check encoding (UTF-8)
   ├── Check row count (≤ 10,000)
   └── Check header row exists

2. COLUMN LEVEL
   ├── All required columns present
   └── No unknown columns (warn only)

3. ROW LEVEL (per row)
   ├── Record Type valid
   ├── Required fields populated
   ├── Values within allowed ranges
   ├── Formats correct (dates, numbers)
   ├── Parent entity exists
   └── No duplicates

4. CROSS-ROW LEVEL
   ├── Parent-child relationships valid
   ├── Campaign names match exactly
   └── No conflicting negatives/positives
```

---

## ✅ PRE-UPLOAD CHECKLIST

- [ ] Record IDs unchanged (or blank for new)
- [ ] All campaigns names case-sensitive match
- [ ] Dates in mm/dd/yyyy format
- [ ] Start dates not in past
- [ ] Bids within currency limits
- [ ] Budgets within currency limits
- [ ] Keywords ≤ 80 chars, no special chars
- [ ] Placement types exact match
- [ ] No keywords in Auto campaigns
- [ ] All parent entities exist or in file
- [ ] No duplicate rows
- [ ] File ≤ 10,000 rows

---

## 🎯 BUSINESS LOGIC: NEGATIVE KEYWORDS

### Two Types of Negatives

| Type | Purpose | Match Type | Ad Group | Status |
|------|---------|------------|----------|--------|
| **Isolation/Harvest** | Block harvested KW in source campaign | `campaign negative exact` ✅ | **BLANK** | `enabled` / `deleted` |
| **Bleeder** | Block bad performers in ad group | `negative exact` or `negative phrase` | **REQUIRED** | `enabled` / `paused` / `archived` |

### Isolation/Harvest Negatives (Campaign-Level)
**When to use:** After promoting a keyword to its own campaign, add as campaign negative to the SOURCE campaign to prevent cannibalization.

```
Record Type: Keyword
Campaign: [SOURCE campaign - where you're ADDING the negative]
Ad Group: [LEAVE BLANK!]
Keyword or Product Targeting: [the harvested keyword]
Match Type: campaign negative exact  ← PREFERRED
Status: enabled
```

⚠️ **Common Mistakes:**
- ❌ Filling in Ad Group (must be blank!)
- ❌ Using "negative exact" instead of "campaign negative exact"
- ❌ Using status "paused" (only enabled/deleted allowed)

### Bleeder Negatives (Ad Group-Level)
**When to use:** Block poor-performing search terms within a specific ad group.

```
Record Type: Keyword
Campaign: [campaign name]
Ad Group: [REQUIRED - specify the ad group]
Keyword or Product Targeting: [the bleeding search term]
Match Type: negative exact  ← or negative phrase
Status: enabled
```

⚠️ **Common Mistakes:**
- ❌ Leaving Ad Group blank (required!)
- ❌ Using "campaign negative" match types

---

## 💰 BUSINESS LOGIC: BID UPDATES

### Required Fields for Bid Updates
| Field | Required | Notes |
|-------|----------|-------|
| Record ID | ✅ YES | Must identify existing keyword |
| Campaign ID | ✅ YES | Required for updates |
| Campaign | ✅ YES | Must match exactly |
| Ad Group | ✅ YES | Must match exactly |
| Max Bid | ✅ YES | The new bid value |

### Bid Update Validation
- ✅ Bid within currency limits
- ⚠️ Warning if change > 300% of current bid
- ❌ Error if Record ID missing

```
Record Type: Keyword
Record ID: [existing Record ID from download]
Campaign ID: [existing Campaign ID]
Campaign: [exact campaign name]
Ad Group: [exact ad group name]
Max Bid: [new bid value]
```

---

## 🚫 AUTO-CAMPAIGN RESTRICTIONS

### What You CANNOT Do in Auto Campaigns
| Action | Allowed? | Error |
|--------|----------|-------|
| Add `broad` keywords | ❌ NO | AUTO001 |
| Add `phrase` keywords | ❌ NO | AUTO001 |
| Add `exact` keywords | ❌ NO | AUTO001 |

### What You CAN Do in Auto Campaigns
| Action | Allowed? |
|--------|----------|
| Add campaign negatives | ✅ YES |
| Add ad group negatives | ✅ YES |
| Update bids | ✅ YES |
| Update status | ✅ YES |
| Modify auto-targeting groups | ✅ YES |

### Auto-Targeting Product Targeting IDs
For Auto campaigns, you can adjust these targeting groups:
- `close-match`
- `substitutes`
- `loose-match`
- `complements`

---

## 🏗️ CAMPAIGN CREATION

### Required Fields (ALL must be present)
```
Record Type: Campaign
Record ID: [BLANK - new campaign]
Campaign ID: [BLANK - new campaign]
Campaign: [unique name, max 128 chars]
Campaign Daily Budget: [within currency limits]
Campaign Start Date: [mm/dd/yyyy, not in past]
Campaign Targeting Type: Auto OR Manual
Campaign Status: enabled
Bidding Strategy: [one of three options]
```

### Bidding Strategy Options
- `Fixed Bids`
- `Dynamic bidding (down only)`
- `Dynamic bidding (up and down)`

---

## 🔍 QUICK VALIDATION DECISION TREE

```
Is this a NEGATIVE keyword?
├── YES → Is Ad Group BLANK?
│   ├── YES → Use "campaign negative exact/phrase" (Isolation)
│   └── NO → Use "negative exact/phrase" (Bleeder)
└── NO → Is this a BID UPDATE?
    ├── YES → Record ID required!
    └── NO → Is campaign AUTO targeting?
        ├── YES → Only negatives allowed!
        └── NO → Proceed with standard validation
```

---

## 📋 VALIDATION ERROR QUICK REFERENCE

### Business Logic Errors
| Code | Issue | Fix |
|------|-------|-----|
| ISO001 | Ad Group filled for campaign negative | Clear the Ad Group field |
| ISO002 | Wrong match type for isolation | Use "campaign negative exact" |
| ISO003 | Invalid status for campaign negative | Use "enabled" or "deleted" only |
| BLD001 | Ad Group missing for bleeder | Fill in the Ad Group field |
| BLD002 | Wrong match type for bleeder | Use "negative exact/phrase" |
| BID_UPD001 | Record ID missing for bid update | Download file first, get Record ID |
| AUTO001 | Positive keyword in Auto campaign | Only negatives allowed in Auto |

---

## 🖥️ UI BEHAVIOR FOR RECOMMENDATIONS

### Status Display
| Status | Icon | Checkbox | Row Style | Export |
|--------|------|----------|-----------|--------|
| Valid | ✅ | Enabled, checked | Normal | ✅ |
| Warning | ⚠️ | Enabled, checked | Yellow bg | ✅ |
| Invalid | ❌ | Disabled | Greyed out | ❌ |

### Example Messages

**Valid:**
```
✅ Add "phone case" as negative exact to Ad Group "Main"
```

**Warning:**
```
⚠️ Bid change of 450% exceeds 300%. Current: 0.50, New: 2.75
   [Still exportable - user acknowledges]
```

**Blocked:**
```
❌ Cannot add positive keywords to Auto campaign "Auto - All Products"
   [Checkbox disabled, cannot export]
```

### Streamlit Example
```python
for rec in recommendations:
    col1, col2, col3 = st.columns([1, 8, 2])
    
    with col1:
        st.checkbox("", value=rec.is_selected, 
                   disabled=not rec.is_valid,
                   key=rec.recommendation_id)
    
    with col2:
        st.write(f"{rec.keyword_text} → {rec.match_type}")
    
    with col3:
        if not rec.is_valid:
            st.error(rec.get_status_icon())
        elif rec.warnings:
            st.warning(rec.get_status_icon())
        else:
            st.success(rec.get_status_icon())
```
