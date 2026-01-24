# Color System

> **Source:** [BRAND_GUIDELINES.md](file:///Users/zayaanyousuf/Documents/Amazon%20PPC/saddle/saddle/desktop/BRAND_GUIDELINES.md)  
> Logo colors extracted from `logo.png`

---

## Core Brand Colors (Primary)

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| **Saddle Deep Purple** | `#5B5670` | 91, 86, 112 | Dark section backgrounds, header/hero panels, primary containers |
| **Charcoal Black** | `#0B0B0D` | 11, 11, 13 | Footer bars, high-contrast text, anchor points |
| **Soft White** | `#E9EAF0` | 233, 234, 240 | Primary text on dark, logo wordmark |

---

## Accent Colors (Decision & Signal)

| Name | Hex | RGB | Usage |
|------|-----|-----|-------|
| **Signal Blue** | `#2A8EC9` | 42, 142, 201 | **Primary CTA**, key data points, validated states, incremental bars |
| **Muted Cyan** | `#8FC9D6` | 143, 201, 214 | Secondary indicators, supporting elements. **Never use alone** |

> Logo icon dots use Signal Blue + Muted Cyan

---

## Light Theme Backgrounds

| Element | Hex | Note |
|---------|-----|------|
| App Background | `#F3F4F7` | Soft cool grey (NOT pure white) |
| Sidebar | `#E8EAF0` | Slightly darker |
| Cards | `#FFFFFF` | White allowed inside cards only |
| Off-white (current site) | `#F5F4F0` | Warm neutral |

---

## Typography Colors (Light Theme)

| Type | Hex | Usage |
|------|-----|-------|
| Primary | `#1F2430` | Headings, key metrics, labels |
| Secondary | `#5E6475` | Descriptions, helper copy |
| Muted | `#9A9AAA` | Disabled text, captions |

---

## Neutral Support Palette

| Name | Hex | Usage |
|------|-----|-------|
| Slate Grey | `#9A9AAA` | Subtext, axis labels, secondary UI |
| Light Grey | `#D6D7DE` | Dividers, disabled states |

---

## Status Colors

> **Note:** Brand guidelines suggest avoiding red/green for emotional neutrality. Use desaturated greys for warnings when possible.

| Name | Hex | Usage |
|------|-----|-------|
| Success | `#10B981` | Positive metrics (use sparingly) |
| Danger | `#EF4444` | Negative metrics (use sparingly) |
| Warning | `#F59E0B` | Caution states |

---

## CSS Variables

```css
:root {
  /* Core Brand */
  --saddle-purple: #5B5670;
  --saddle-black: #0B0B0D;
  --saddle-white: #E9EAF0;
  
  /* Accent (CTAs & Signals) */
  --signal-blue: #2A8EC9;
  --muted-cyan: #8FC9D6;
  
  /* Neutral */
  --slate-grey: #9A9AAA;
  --light-grey: #D6D7DE;
  
  /* Light Theme Backgrounds */
  --bg-app: #F3F4F7;
  --bg-sidebar: #E8EAF0;
  --bg-card: #FFFFFF;
  
  /* Typography */
  --text-primary: #1F2430;
  --text-secondary: #5E6475;
  --text-muted: #9A9AAA;
}
```

---

## Guardrails

- ❌ NO Emerald/Green as primary accent (not brand-aligned)
- ❌ NO neon colors or glow effects
- ❌ NO purple gradients inside logo
- ❌ NO competing CTAs (one primary CTA per screen)
- ✅ Signal Blue `#2A8EC9` for all primary CTAs
- ✅ Saddle Purple `#5B5670` for dark sections
- ✅ Cool greys for backgrounds, not pure white
- ✅ Evidence-first, not emotional (muted status colors)
