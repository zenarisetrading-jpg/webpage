# Parallel Orchestration Plan - 8 Pages
## With Strict Guardrails: Minimal, Suave, No Fabrication

---

## Phase 0: Backup Current Site (CRITICAL)

**Before anything else:**

1. **Save complete copy of current live site**
   ```
   /backup/
   ├── index.html (original)
   ├── agencies.html (original)
   ├── features.html (original)
   ├── compare.html (original)
   ├── about.html (original)
   ├── faq.html (original)
   ├── how-it-works.html (original)
   ├── audit.html (original)
   ├── styles.css (original)
   ├── agencies-styles.css (original)
   ├── script.js (original)
   ├── audit.js (original)
   └── all assets/
   ```

2. **Version control**
   - Tag current version as `v1.0-before-redesign`
   - No touching production until full approval
   - All work happens in `/redesign/` folder

3. **Rollback plan**
   - If anything goes wrong, instant revert to backup
   - No messing around with live site

---

## Overview: Parallel Build Structure

```
Phase 1: Master Agent (Design System) → 6 hours
         │
         ├─→ Creates CORRECTED design system
         ├─→ Extracts ALL content from current HTML
         ├─→ Creates 8 page briefs (content only, no fabrication)
         └─→ Sets up guardrail templates
         
Phase 2: 8 Agents Work Simultaneously → 12-16 hours
         │
         ├─→ Agent 1: Homepage
         ├─→ Agent 2: Features
         ├─→ Agent 3: Agencies
         ├─→ Agent 4: Compare
         ├─→ Agent 5: About
         ├─→ Agent 6: FAQ
         ├─→ Agent 7: How It Works
         └─→ Agent 8: Audit Tool
         
Phase 3: Integration & Review → 4-6 hours
         │
         └─→ Master combines all pages
             Checks consistency
             Your final approval
             
Total: ~24 hours elapsed, 2 hours of your time
```

---

## Phase 1: Master Agent - Foundation Setup

### Master Agent Responsibilities:

**Task 1: Create Corrected Design System (3 hours)**
```
/design-system/
├── CORRECTED_DESIGN_PRINCIPLES.md (already created)
├── colors.md (muted, professional palette)
├── typography.md (existing sizes, slight metric bump)
├── spacing.md (reasonable padding/margins)
├── components.md (buttons, cards, badges - minimal)
└── layout-patterns.md (subtle asymmetry, clean grids)
```

**Content:**
- Color palette: Emerald-600, Slate-800, White/Slate-50
- Typography: Max 48px for metrics, keep existing otherwise
- Components: Clean, minimal, no flashy effects
- Layouts: Subtle 60/40 splits, NOT extreme
- NO: Neon, glow, heavy gradients, giant fonts

**Task 2: Extract Content from Current Site (2 hours)**
```
/content-extraction/
├── homepage-content.json (every word, number, link)
├── agencies-content.json
├── features-content.json
├── compare-content.json
├── about-content.json
├── faq-content.json
├── how-it-works-content.json
└── audit-content.json
```

**Process:**
- Parse every current HTML file
- Extract ALL text, ALL numbers, ALL links
- No interpretation, just extraction
- This becomes the ONLY source of truth for content

**Task 3: Create 8 Page Briefs (1 hour)**

Each brief contains:
```markdown
# [PAGE NAME] Brief

## Content (From Extraction - NO CHANGES ALLOWED)
[Exact content from current HTML]

## Current Structure
[Section breakdown of current page]

## Redesign Instructions
1. Layout: [Specific layout for each section]
2. Styling: [Reference to design system]
3. Content usage: EXACT content from extraction, no additions

## Forbidden
- Adding any stats/numbers not in extraction
- Creating new sections
- Fabricating testimonials/proof
- Giant fonts (>48px)
- Neon/flashy styling
- AI design tells
```

---

## Phase 2: 8 Agents Work Simultaneously

### Agent Instructions Template

Each agent receives:

1. **Design system** (CORRECTED_DESIGN_PRINCIPLES.md)
2. **Page brief** (content + structure requirements)
3. **Content extraction** (JSON with exact content)
4. **Guardrails** (strict rules)

### Strict Guardrails for Each Agent

```markdown
# CRITICAL RULES (BREAK THESE = REJECT WORK)

## Content Rules:
1. Use ONLY content from provided content extraction JSON
2. DO NOT add any numbers, stats, or claims not in extraction
3. DO NOT create testimonials or social proof
4. DO NOT fabricate case studies or results
5. Every word must trace to original HTML

## Design Rules:
1. Font sizes: Max 48px for metrics, keep existing otherwise
2. Colors: Emerald-600, Slate-800, White/Slate-50 (NO neon)
3. Backgrounds: Primarily white/light gray, max 1 dark section
4. Effects: NO glow, NO pulsing, NO heavy gradients
5. Animations: Simple hover states only (0.2s transitions)

## Layout Rules:
1. Asymmetry: Subtle 60/40 or 55/45 splits (NOT extreme)
2. Spacing: 80px section padding (NOT 128px)
3. Grids: Simple 2-col or 3-col (NO bento grids)
4. White space: Generous but not excessive
5. NO sticky scroll, NO floating elements

## Validation:
Before submitting, check:
☐ All content from extraction JSON?
☐ Font sizes < 48px?
☐ Colors muted/professional?
☐ NO neon/glow/flash?
☐ Looks human-designed?
☐ Minimal and suave?

If ANY "No" → Fix before submitting.
```

---

## Phase 2 Details: Agent Assignments

### Agent 1: Homepage

**Sections to redesign:**
1. Hero - asymmetric 60/40, keep existing copy
2. Problem metrics - 3 cards in row, clean presentation
3. Product showcase - video left (60%), 4 cards right (40%)
4. Process - 3 steps, clean layout
5. Testimonial - simple quote block
6. Final CTA

**Forbidden:**
- Sticky scroll
- Floating metrics
- Zigzag layouts
- Giant numbers (>48px)
- Timeline connectors
- Fabricated stats

**Output:** `homepage-redesign.html`

---

### Agent 2: Features

**Sections to redesign:**
(Based on current features.html structure)
- Feature list with clean cards
- Product screenshots/demos
- Benefit explanations

**Forbidden:**
- Adding features not in current site
- Fabricating metrics
- Over-designing showcases

**Output:** `features-redesign.html`

---

### Agent 3: Agencies

**Sections to redesign:**
1. Hero - clean headline, no giant question
2. Problem scenarios - 3 cards, existing content only
3. Benefits - clean grid, existing 6 benefits
4. White label section
5. Final CTA

**Forbidden:**
- 34% retention stat (fabricated)
- $12k upsell stat (fabricated)
- 8 hours saved stat (fabricated)
- Before/after with made-up data
- Results timeline
- Any ROI numbers not in original

**Output:** `agencies-redesign.html`

---

### Agent 4: Compare

**Sections to redesign:**
- Comparison content from current site
- Feature comparison tables
- Differentiation points

**Forbidden:**
- Adding competitors not mentioned
- Fabricating feature comparisons
- Over-designing comparison UI

**Output:** `compare-redesign.html`

---

### Agent 5: About

**Sections to redesign:**
- Founder story
- Mission/values
- Company background

**Forbidden:**
- Adding stats not in current site
- Fabricating milestones
- Creating timeline if doesn't exist

**Output:** `about-redesign.html`

---

### Agent 6: FAQ

**Sections to redesign:**
- Q&A pairs from current site
- Organized by category

**Forbidden:**
- Adding questions not in current site
- Fabricating answers
- Over-designed accordion

**Output:** `faq-redesign.html`

---

### Agent 7: How It Works

**Sections to redesign:**
- Process walkthrough
- Step-by-step guide

**Forbidden:**
- Adding steps not in current site
- Fabricating screenshots
- Over-designed timeline

**Output:** `how-it-works-redesign.html`

---

### Agent 8: Audit Tool

**Sections to redesign:**
- Upload interface
- Quiz/questions (existing only)
- Results display

**Forbidden:**
- Changing quiz questions
- Adding metrics to results
- Over-designing interface

**Output:** `audit-redesign.html`

---

## Phase 3: Integration & Review

### Master Agent Tasks:

**1. Combine All Pages (2 hours)**
- Merge 8 HTML files
- Ensure navigation works
- Check cross-page consistency
- Verify all links

**2. Consistency Check (2 hours)**
- Color palette consistent?
- Typography consistent?
- Spacing consistent?
- Component styling consistent?

**3. Content Audit (1 hour)**
- Compare every page to original HTML
- Verify ZERO fabricated content
- Check all numbers trace to original
- Flag any discrepancies

**4. Design Audit (1 hour)**
- Font sizes all < 48px?
- NO neon/glow/flashy?
- Minimal and suave?
- Human-designed feel?

### Quality Gates:

Before presenting to you:
```
☐ All 8 pages complete
☐ Navigation working
☐ Zero fabricated content verified
☐ Design principles followed
☐ No AI tells present
☐ Passes all guardrails
☐ Backup of original safe
```

---

## Your Review Process (Phase 4)

**Step 1: Quick Visual Scan**
- Does it FEEL minimal and suave?
- Any AI tells jump out?
- Font sizes reasonable?

**Step 2: Content Verification**
- Spot check 3-4 sections
- Compare to original HTML
- Any fabricated content?

**Step 3: Design Check**
- Colors muted/professional?
- Spacing reasonable?
- Layouts clean?

**Decision Points:**
- ✓ Approve → Deploy
- ⚠️ Minor fixes → Agent adjusts
- ✗ Major issues → Restart with clearer brief

---

## Timeline

**Day 1:**
- Hour 0-6: Master creates foundation
- Hour 6-22: 8 agents work in parallel
- Hour 22-24: Integration begins

**Day 2:**
- Hour 0-6: Integration completes
- Hour 6-8: Your review
- Hour 8+: Adjustments if needed

**Total: ~30 hours elapsed, 2-3 hours of your time**

---

## Risk Mitigation

### Risk 1: Agent Fabricates Content
**Prevention:**
- Content extraction JSON as single source of truth
- Explicit "NO FABRICATION" in every brief
- Content audit before presenting to you

**Mitigation:**
- If caught, immediate rework from that agent
- Master re-checks all pages

### Risk 2: Agent Over-Designs
**Prevention:**
- CORRECTED_DESIGN_PRINCIPLES in every brief
- Examples of "good" vs "bad"
- Validation checklist in instructions

**Mitigation:**
- Design audit catches this
- Agent adjusts to minimal aesthetic

### Risk 3: Inconsistency Across Pages
**Prevention:**
- Shared design system
- Master agent creates component library
- Regular consistency checks

**Mitigation:**
- Master agent's integration phase fixes

### Risk 4: Original Site Corrupted
**Prevention:**
- Phase 0: Complete backup
- Work in separate /redesign/ folder
- No touching production

**Mitigation:**
- Instant rollback from backup
- Version control

---

## Tools & Setup

### Directory Structure:
```
/saddle-redesign/
├── /backup/ (original site - DON'T TOUCH)
├── /design-system/ (master creates)
├── /content-extraction/ (master creates)
├── /page-briefs/ (master creates)
├── /agent-outputs/
│   ├── homepage-redesign.html
│   ├── features-redesign.html
│   ├── agencies-redesign.html
│   ├── compare-redesign.html
│   ├── about-redesign.html
│   ├── faq-redesign.html
│   ├── how-it-works-redesign.html
│   └── audit-redesign.html
├── /integrated/ (master combines here)
└── /final/ (after your approval)
```

### Communication:
- Each agent outputs to their file
- Master monitors progress
- Flags issues immediately
- You review only final integrated version

---

## Success Criteria

Before considering this done:

**Content:**
- ☐ Zero fabricated stats/numbers
- ☐ All content traces to original HTML
- ☐ No added testimonials/social proof
- ☐ No case studies created

**Design:**
- ☐ Font sizes < 48px (except where noted)
- ☐ Muted, professional color palette
- ☐ NO neon, glow, or flashy effects
- ☐ Minimal and suave aesthetic
- ☐ NO AI design tells

**Technical:**
- ☐ All 8 pages complete
- ☐ Navigation works
- ☐ Responsive design
- ☐ Performance maintained

**Process:**
- ☐ Original site safely backed up
- ☐ No production changes
- ☐ Your approval received

---

## Next Steps to Execute

**If you approve this plan:**

1. I'll create the Master Agent setup files:
   - Design system documents
   - Content extraction scripts
   - 8 page briefs
   - Agent instruction templates

2. You review the foundation (30 min)

3. We launch the 8 agents in parallel

4. 24 hours later: integrated site for your review

**Sound good?**
