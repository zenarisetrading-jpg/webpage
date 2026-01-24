# CORRECTED Design Principles
## Minimal, Suave, Professional - No AI Tells

---

## Critical Guardrails (NON-NEGOTIABLE)

### 1. Font Sizes - Subtle Hierarchy Only
**FORBIDDEN:**
- ✗ Giant typography (7xl, 8xl, 9xl - 72px+)
- ✗ Display numbers dominating the page
- ✗ Typography as primary visual element

**ALLOWED:**
- ✓ Keep existing font sizes as base
- ✓ Metrics can be 1.5-2x larger than body text (max 48px)
- ✓ Subtle emphasis through weight, not size
- ✓ Hierarchy through spacing and color, not scale

**Examples:**
```
Body text: 16px (existing)
Metric/number: 32-48px (slight bump, not giant)
Headline: 36-48px (existing range)
Subhead: 20-24px (existing)
```

---

### 2. Content - Zero Fabrication
**FORBIDDEN:**
- ✗ Any stats not in original HTML
- ✗ Testimonials not provided by user
- ✗ Case studies or ROI numbers (34%, $12k, etc.)
- ✗ Social proof that doesn't exist
- ✗ "Agency results" sections with made-up data
- ✗ ANY content not explicitly in current site

**ALLOWED:**
- ✓ Exact content from current HTML files
- ✓ Restructure existing content
- ✓ Reformat existing content
- ✓ Break up or combine existing sections
- ✓ Change visual presentation of existing content

**Rule:** If it's not in the current HTML, it doesn't go in the redesign.

---

### 3. Aesthetic - Minimal & Suave (No AI Tells)
**FORBIDDEN:**
- ✗ Neon colors (bright emerald-400, cyan, hot pink)
- ✗ Heavy gradients everywhere
- ✗ Glow effects / pulsing animations
- ✗ Bloomberg Terminal aesthetic
- ✗ "Techie" dashboard vibes
- ✗ Overly dramatic dark sections
- ✗ Floating elements everywhere
- ✗ Heavy backdrop blur
- ✗ Rainbow accent colors

**ALLOWED:**
- ✓ Clean, professional color palette
- ✓ Subtle gradients (if needed)
- ✓ White/light gray backgrounds primarily
- ✓ Dark sections sparingly (1-2 max)
- ✓ Emerald-600 as accent (not neon)
- ✓ Generous white space
- ✓ Simple, clean transitions
- ✓ Professional shadows (subtle)

**Color Palette (Revised):**
```
Primary: Emerald-600 #059669 (muted, professional)
Dark: Slate-800 #1e293b (not pure black)
Light: White / Slate-50 #f8fafc
Text: Slate-900 #0f172a
Secondary: Slate-600 #475569
Accent (sparingly): Blue-600 #2563eb

AVOID: Bright emerald-400, neon anything, excessive gradients
```

---

## What "Data-Driven Narrative Flow" Actually Means (Corrected)

### NOT This:
- Giant numbers everywhere
- Dashboard-first product showcase
- Flashy animations
- Heavy visual effects

### Actually This:
1. **Logical content flow** - Information reveals in order that makes sense
2. **Subtle asymmetry** - Not everything centered, but not extreme
3. **Breathing room** - White space between sections
4. **Visual hierarchy** - Through spacing and weight, not size
5. **Clean transitions** - Section to section feels natural
6. **Minimal styling** - Let content speak, not design

---

## Design Patterns (Corrected)

### ✓ GOOD: Asymmetric Layouts (Subtle)
```
60/40 or 55/45 splits (not extreme)
Text slightly off-center
Images on one side, text on other
NOT: Zigzag patterns, diagonal timelines
```

### ✓ GOOD: Clean Cards
```
White cards with subtle border
Clean shadows (not heavy)
Grouped logically
Minimal padding (not excessive)
```

### ✓ GOOD: Section Breaks
```
Alternating white/light gray backgrounds
One dark section for contrast (optional)
Generous vertical spacing
Clear section boundaries
```

### ✗ BAD: Over-designed Elements
```
Floating metric cards around product
Sticky scroll with progressive reveals
Timeline dots and connectors
Pulsing glows
Heavy gradients
```

---

## Typography (Corrected)

### Font Scale:
```
h1 (Hero): 42-48px, Bold (existing)
h2 (Section): 32-36px, Bold (existing)
h3 (Card title): 20-24px, Semibold (existing)
Body: 16px, Regular (existing)
Small: 14px, Regular (existing)

Metrics (slight bump): 36-48px, Bold
NOT 72px, 96px, 128px - those are eye sores
```

### Font Weights:
```
Regular: 400 (body)
Medium: 500 (labels)
Semibold: 600 (subheads)
Bold: 700 (headlines, metrics)

AVOID: Extrabold (800), Black (900)
```

---

## Layout Principles (Corrected)

### 1. Container Widths
```
Max width: 1280px (existing)
Section padding: 80px vertical (not 128px)
Content padding: 40-60px (not excessive)
```

### 2. Grid Systems
```
2-column: 50/50 or 60/40 (subtle asymmetry)
3-column: Equal or slight variation
4-column: Equal widths

AVOID: Bento grids with wild size variations (8+4, 5+7)
```

### 3. Spacing
```
Section gaps: 80-120px
Card gaps: 24-32px
Element gaps: 16-24px

AVOID: Excessive white space, overly tight spacing
```

---

## Animation & Interaction (Minimal)

### ✓ ALLOWED:
- Hover state transitions (0.2s)
- Simple fade-ins on scroll (optional)
- Button hover effects (subtle)
- Link underlines on hover

### ✗ FORBIDDEN:
- Sticky scroll sections
- Progressive metric reveals
- Pulsing/glowing elements
- Complex scroll-triggered animations
- Floating cards that appear on scroll
- Count-up number animations

---

## Content Structure (Existing Only)

### Homepage Sections (From Current HTML):
1. Hero - headline + CTA
2. Problem - 3 metrics ($2,847, 23 winners, 184 targets)
3. Product showcase - "This is what knowing looks like" + video
4. 4 feature cards (impact, ROAS, decision score, trust)
5. Process - 3 steps (upload, review, see results)
6. Testimonial quote (agency owner)
7. Founder story link
8. Final CTA

**DO NOT ADD:**
- Case studies
- ROI stats
- Social proof metrics
- Agency results timelines
- Client retention numbers
- Any other fabricated content

### Agencies Page Sections (From Current HTML):
1. Hero - headline + subtitle
2. The renewal call problem (3 scenarios)
3. Benefits grid (6 benefits)
4. White label section
5. Pricing preview (if exists)
6. Final CTA

**DO NOT ADD:**
- 34% retention lift
- $12k upsell numbers
- 8 hours saved stats
- Before/after comparisons with made-up data
- Timeline of results
- Any fabricated proof points

---

## Examples: What Changes

### Homepage Section 3 (Product Showcase)

**WRONG (What I Did):**
```
- Sticky video section
- 4 floating metric cards
- Dark gradient background
- Glow effects
- Progressive scroll reveals
```

**RIGHT (What to Do):**
```
- Split layout: Video left (60%), Cards right (40%)
- Clean white/light gray background
- Video in simple container (no MacBook mockup)
- 4 cards stacked on right (existing content)
- Subtle shadows, no glows
- No scroll effects
```

### Agencies Section 2 (Problem)

**WRONG (What I Did):**
```
- 3-Act structure with dramatic color coding
- Red/Dark/Green borders
- Made-up dialogue format
- "Client thinks:" fabrications
```

**RIGHT (What to Do):**
```
- 3 scenario cards (existing content)
- Clean white cards with subtle borders
- Existing text only
- No dramatic color coding
- Simple, professional presentation
```

---

## Visual References (What to Aim For)

### ✓ GOOD Examples:
- Stripe.com - Clean, minimal, professional
- Linear.app - Simple, suave, effective
- Pitch.com - Modern but not flashy
- Notion.com - Clean cards, subtle asymmetry

### ✗ BAD Examples (Avoid):
- My mockups (too flashy, giant fonts, fabricated content)
- Overly "designed" AI-generated sites
- Dashboard-heavy designs
- Anything with neon or heavy effects

---

## Summary: The Real Brief

**Keep:**
- Existing content ONLY
- Existing font sizes (slight metric bump allowed)
- Clean, professional aesthetic
- White/light backgrounds primarily
- Subtle asymmetry where appropriate

**Change:**
- Layout structure (asymmetric splits)
- Section organization (better flow)
- Visual hierarchy (through spacing/weight)
- Section transitions (cleaner breaks)
- Card arrangements (logical grouping)

**Avoid:**
- Giant typography
- Fabricated content
- AI design tells (neon, glow, flashy)
- Overly dramatic effects
- Heavy animations

---

## Validation Checklist (Every Section)

Before finalizing any section, ask:
1. ☐ Are ALL numbers/stats from original HTML?
2. ☐ Are font sizes reasonable (< 48px)?
3. ☐ Is the color palette muted/professional?
4. ☐ Is there NO neon/glow/flash?
5. ☐ Could this pass as human-designed?
6. ☐ Is it minimal and suave?
7. ☐ Would user approve this aesthetic?

If ANY answer is "No" → Redesign that element.

---

This is the corrected brief for orchestration.
