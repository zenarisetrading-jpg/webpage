  # SADDLE – Brand Color Palette & Usage Guidelines

  ## 1. Core Brand Colors (Primary)

  ### Saddle Deep Purple (Primary Base)
  - **Hex:** `#5B5670`
  - **RGB:** 91, 86, 112
  - **Usage:** App background (dark sections), header/hero panels, primary containers
  - **Meaning:** Calm authority, technical depth, trust

  ### Charcoal Black (Anchor / Contrast)
  - **Hex:** `#0B0B0D`
  - **RGB:** 11, 11, 13
  - **Usage:** Footer bars, login base strip, text on light backgrounds
  - **Meaning:** Serious, grounded, enterprise-grade

  ### Soft White (Primary Text / Logo Type)
  - **Hex:** `#E9EAF0`
  - **RGB:** 233, 234, 240
  - **Usage:** Logo wordmark, primary text on dark backgrounds
  - **Meaning:** Clarity, restraint, premium feel

  ---

  ## 2. Accent Colors (Decision & Signal)

  ### Signal Blue (Decision Accent)
  - **Hex:** `#2A8EC9`
  - **RGB:** 42, 142, 201
  - **Usage:** Key data points, incremental bars, "Validated / Confirmed" states
  - **Rule:** Use only where decisions or validation are shown

  ### Muted Cyan (Secondary Signal)
  - **Hex:** `#8FC9D6`
  - **RGB:** 143, 201, 214
  - **Usage:** Secondary indicators, supporting dots/connectors
  - **Rule:** Never use alone — always paired with Signal Blue

  ---

  ## 3. Neutral Support Palette

  | Color | Hex | Usage |
  |-------|-----|-------|
  | Slate Grey | `#9A9AAA` | Subtext, axis labels, secondary UI copy |
  | Light Grey | `#D6D7DE` | Dividers, disabled states, inactive icons |

  ---

  ## 4. Logo Usage Rules

  ### ✅ Allowed
  - Logo on solid dark backgrounds only
  - Wordmark in Soft White
  - Icon dots in Signal Blue / Muted Cyan only

  ### ❌ Not Allowed
  - Gradients inside the logo
  - Shadow effects
  - Changing dot colors
  - Using logo on busy charts or images
  - Placing logo on pure white without a container

  ---

  ## 5. UI Color Hierarchy

  | Purpose | Color |
  |---------|-------|
  | Primary action / insight | Signal Blue `#2A8EC9` |
  | Neutral data | Soft White / Slate Grey |
  | Background | Saddle Deep Purple `#5B5670` |
  | Warnings / negatives | Desaturated greys only (no red) |

  > **Note:** No red or green by default. This reinforces "evidence-first, not emotional" positioning.

  ---

  ## 6. Typography

  - **Headlines:** Inter / Satoshi / SF Pro (SemiBold)
  - **Body:** Inter / SF Pro (Regular)
  - **Numbers:** Same family, tabular numerals enabled

  > No decorative fonts. No rounded playful typefaces.

  ---

  ## 7. Brand Personality

  - ❌ Not flashy AI
  - ❌ Not growth-hack SaaS
  - ✅ Calm, confident, operator-built
  - ✅ Trust before automation
  - ✅ Decisions over dashboards

  ---

  ## Quick Reference (CSS Variables)

  ```css
  :root {
    /* Core */
    --saddle-purple: #5B5670;
    --saddle-black: #0B0B0D;
    --saddle-white: #E9EAF0;
    
    /* Accent */
    --signal-blue: #2A8EC9;
    --muted-cyan: #8FC9D6;
    
    /* Neutral */
    --slate-grey: #9A9AAA;
    --light-grey: #D6D7DE;
  }
  ```

  ---

  ## 8. Light Theme Specification

  > "A calm audit room, not a SaaS landing page"

  ### Global Backgrounds
  | Element | Color | Note |
  |---------|-------|------|
  | App background | `#F3F4F7` | Soft cool grey (NOT pure white) |
  | Sidebar | `#E8EAF0` | Slightly darker |
  | Cards | `#FFFFFF` | White allowed only inside cards |

  ### Typography
  | Type | Color | Usage |
  |------|-------|-------|
  | Primary | `#1F2430` | Headings, key metrics, labels |
  | Secondary | `#5E6475` | Descriptions, helper copy |
  | Muted | `#9A9AAA` | Disabled text |

  ### Tables
  | Element | Color |
  |---------|-------|
  | Background | `#FFFFFF` |
  | Header row | `#F7F8FB` |
  | Row divider | `#ECEEF4` |
  | Hover state | `#F1F4FA` |
  | Selected row | `#EAF3FB` |

  ### Cards & Borders
  - **Card background:** `#FFFFFF`
  - **Border:** `1px solid #E2E4EC`
  - **Shadow:** `0 1px 2px rgba(0,0,0,0.04)` (very subtle)

  ### CTAs
  | Type | Style |
  |------|-------|
  | Primary | Background `#2A8EC9`, Text `#FFFFFF` |
  | Secondary | Border `#2A8EC9`, Text `#2A8EC9`, Transparent bg |
  | Tertiary | Text only `#5E6475` |

  > ⚠️ One primary CTA per screen. No competing buttons.

  ### Charts
  - **Axis lines:** `#E6E8F0`
  - **Labels:** `#5E6475`
  - **Canvas:** Transparent (inherits card white)

