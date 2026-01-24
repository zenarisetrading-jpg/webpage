# Audit Page Brief

## Current Structure
- Hero with headline
- Quiz section (5 questions with progress)
- Results section (score, opportunity, breakdown, improvements)
- Upload section (dropzone with instructions)

## Extracted Content Reference
- File: `/content-extraction/audit-content.json`
- ALL content must come from this file
- NO additions allowed

## Redesign Layout

### Section 1: Hero
**Current:** Title and subtitle
**Redesign:** Clean centered, minimal  
**Styling:** H1 at 36px, subtitle in Slate-600

### Section 2: Quiz
**Current:** Card with progress bar, question grid
**Redesign:** Card with max-width 700px, clean progress indicator  
**Styling:**
- Progress bar: Emerald-600 fill
- Questions: Radio-style options
- CTA: Full-width Emerald-600 button

### Section 3: Results
**Current:** Multi-card grid (score, opportunity, breakdown, improvements)
**Redesign:** 2-column layout: score left, details right  
**Styling:**
- Score: Large 48px number with color indicator
- Cards: White with subtle borders
- Success/danger colors for positive/negative values

### Section 4: Upload
**Current:** Dropzone with file input and instructions
**Redesign:** Centered card, dashed border dropzone  
**Styling:**
- Dropzone: Dashed 2px border, Slate-300
- Hover: Emerald-600 border
- Instructions: Numbered list, clean

## Strict Guardrails
☐ Use ONLY content from extraction JSON
☐ Max font size: 48px
☐ Colors: Emerald-600, Slate-800, White/Slate-50
☐ NO fabrication, NO neon, NO glow, NO flashy
☐ Minimal & suave aesthetic

## Validation Checklist
Before submission:
☐ All content from extraction?
☐ Font sizes < 48px?
☐ Colors muted?
☐ NO AI tells?
☐ Looks human-designed?
