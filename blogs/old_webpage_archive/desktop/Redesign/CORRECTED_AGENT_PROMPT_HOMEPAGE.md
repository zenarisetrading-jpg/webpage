# CORRECTED AGENT PROMPT - Homepage Rebuild

## Copy and paste this into Claude Code:

```
HOMEPAGE REBUILD - Data-Driven Narrative Flow

CRITICAL CONTEXT:
The first attempt failed because you only changed colors. User feedback: "It's just a redo of my existing site in different color... nothing has changed"

⚠️ CRITICAL RULE: 1:1 CONTENT MAPPING
────────────────────────────────────
ALL content must be EXACTLY copied from original HTML.
- NO paraphrasing (if original says "50+ clicks" → use "50+ clicks" not "over 50 clicks")
- NO rewording (if original says "actually made money" → use "actually made money" not "really made money")
- NO creative interpretation
- Character-by-character match required

HOW TO DO THIS:
1. Open /desktop/Redesign/index.html (original file)
2. Find exact text you need
3. Copy-paste it (don't retype)
4. Verify it matches exactly

WHAT CHANGES: Layout, positioning, styling, arrangement
WHAT DOESN'T CHANGE: Any text content whatsoever
────────────────────────────────────

READ THIS FIRST:
/desktop/Redesign/CORRECTED_BRIEF_NARRATIVE_FLOW.md

This explains what "data-driven narrative flow" actually means.

WHAT USER WANTS:
✓ Same exact CONTENT (every word, number, link)
✓ Completely DIFFERENT LAYOUTS (asymmetric, zigzag, varied rhythm)
✓ Non-linear flow (breaks section-section-section monotony)
✓ Reasonable font sizes (max 48px, use 36-42px for metrics)
✓ Minimal aesthetic (emerald-600, no neon, no flashy)

WHAT YOU DID WRONG LAST TIME:
✗ Just changed colors
✗ Kept exact same centered layouts
✗ All sections look identical
✗ Predictable rhythm
✗ BORING

WHAT TO DO THIS TIME:
✓ Keep content exact
✓ Change EVERY section layout
✓ Use asymmetric splits (60/40, 55/45)
✓ Alternate backgrounds (white, light gray, dark)
✓ Stagger elements (not rigid grids)
✓ Create visual flow

─────────────────────────────────────────

YOUR TASK: Rebuild homepage.html

SECTION-BY-SECTION INSTRUCTIONS:

SECTION 1: HERO
───────────────
Current: Centered everything
Redesign: Asymmetric 60/40 split

Layout:
```html
<section style="padding: 100px 0; background: white;">
  <div class="container" style="display: grid; grid-template-columns: 60% 38%; gap: 60px; align-items: center;">
    
    <!-- LEFT 60%: Text content -->
    <div>
      <h1 style="font-size: 48px; font-weight: bold; line-height: 1.1; margin-bottom: 24px;">
        [Exact headline from content extraction]
      </h1>
      <p style="font-size: 20px; color: #475569; margin-bottom: 32px; line-height: 1.6;">
        [Exact subtitle from content extraction]
      </p>
      <a href="audit.html" style="display: inline-block; background: #059669; color: white; padding: 16px 32px; border-radius: 8px; font-weight: 600; text-decoration: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        [Exact CTA text]
      </a>
      <p style="font-size: 14px; color: #64748b; margin-top: 16px;">
        [Exact proof text]
      </p>
    </div>
    
    <!-- RIGHT 40%: Floating metrics card -->
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 32px; box-shadow: 0 8px 24px rgba(0,0,0,0.08);">
      <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 12px;">
        Your Current Situation
      </div>
      <div style="font-size: 36px; font-weight: bold; color: #ef4444; margin-bottom: 8px;">
        $2,847
      </div>
      <div style="font-size: 16px; font-weight: 600; color: #1f2937; margin-bottom: 4px;">
        burning/mo
      </div>
      <div style="font-size: 14px; color: #64748b; margin-bottom: 24px;">
        [Description from content]
      </div>
      <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;">
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; text-align: center;">
        <div>
          <div style="font-size: 28px; font-weight: bold; color: #059669;">23</div>
          <div style="font-size: 12px; color: #64748b;">winners buried</div>
        </div>
        <div>
          <div style="font-size: 28px; font-weight: bold; color: #f59e0b;">184</div>
          <div style="font-size: 12px; color: #64748b;">targets broken</div>
        </div>
      </div>
    </div>
    
  </div>
</section>
```

SECTION 2: PROBLEM METRICS
───────────────────────────
Current: 3 equal cards in a row
Redesign: Staggered cascade with connecting flow

Layout:
```html
<section style="padding: 80px 0; background: #f8fafc;">
  <div class="container" style="max-width: 1000px;">
    
    <!-- Staggered cards -->
    <div style="position: relative;">
      
      <!-- Card 1 - Top left -->
      <div style="background: white; border-radius: 16px; padding: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom: 32px; margin-left: 0;">
        <div style="font-size: 42px; font-weight: bold; color: #ef4444; margin-bottom: 12px;">
          $2,847
        </div>
        <div style="font-size: 20px; font-weight: 600; color: #1f2937; margin-bottom: 8px;">
          burning/mo
        </div>
        <p style="color: #64748b; line-height: 1.6;">
          [Description]
        </p>
      </div>
      
      <!-- Card 2 - Middle, offset right -->
      <div style="background: white; border-radius: 16px; padding: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom: 32px; margin-left: 80px;">
        <div style="font-size: 42px; font-weight: bold; color: #059669; margin-bottom: 12px;">
          23 winners
        </div>
        <div style="font-size: 20px; font-weight: 600; color: #1f2937; margin-bottom: 8px;">
          buried
        </div>
        <p style="color: #64748b; line-height: 1.6;">
          [Description]
        </p>
      </div>
      
      <!-- Card 3 - Bottom, offset more right -->
      <div style="background: white; border-radius: 16px; padding: 40px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-left: 160px;">
        <div style="font-size: 42px; font-weight: bold; color: #f59e0b; margin-bottom: 12px;">
          184 targets
        </div>
        <div style="font-size: 20px; font-weight: 600; color: #1f2937; margin-bottom: 8px;">
          need fixes
        </div>
        <p style="color: #64748b; line-height: 1.6;">
          [Description]
        </p>
      </div>
      
    </div>
    
    <!-- Accusation line - centered -->
    <p style="text-align: center; font-size: 24px; font-weight: 600; color: #1f2937; margin: 60px 0 32px;">
      [Exact accusation text]
    </p>
    
    <div style="text-align: center;">
      <a href="audit.html" style="display: inline-block; background: #059669; color: white; padding: 16px 32px; border-radius: 8px; font-weight: 600; text-decoration: none;">
        [Exact CTA]
      </a>
    </div>
    
  </div>
</section>
```

SECTION 3: PRODUCT SHOWCASE
────────────────────────────
Current: Centered video, 4 cards below in grid
Redesign: Split 55/45, video left, cards stacked right

Layout:
```html
<section style="padding: 100px 0; background: white;">
  <div class="container">
    
    <!-- Full-width headline -->
    <h2 style="font-size: 42px; font-weight: bold; text-align: center; margin-bottom: 60px;">
      This is what <span style="color: #059669;">knowing</span> looks like.
    </h2>
    
    <!-- Split layout -->
    <div style="display: grid; grid-template-columns: 55% 43%; gap: 60px; align-items: start;">
      
      <!-- LEFT: Video -->
      <div>
        <div style="border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.12);">
          <video autoplay loop muted playsinline style="width: 100%; display: block;">
            <source src="mock-dash.mp4" type="video/mp4">
          </video>
        </div>
        <p style="text-align: center; color: #64748b; margin-top: 24px; font-size: 16px;">
          Every number traces back to the decision that caused it.
        </p>
      </div>
      
      <!-- RIGHT: Cards stacked -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        
        <!-- Card 1 -->
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px;">
          <h3 style="font-size: 20px; font-weight: 600; color: #1f2937; margin-bottom: 12px;">
            [Card 1 title]
          </h3>
          <p style="color: #64748b; font-size: 15px; line-height: 1.6;">
            [Card 1 content]
          </p>
        </div>
        
        <!-- Card 2 -->
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px;">
          <h3 style="font-size: 20px; font-weight: 600; color: #1f2937; margin-bottom: 12px;">
            [Card 2 title]
          </h3>
          <p style="color: #64748b; font-size: 15px; line-height: 1.6;">
            [Card 2 content]
          </p>
        </div>
        
        <!-- Card 3 -->
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px;">
          <h3 style="font-size: 20px; font-weight: 600; color: #1f2937; margin-bottom: 12px;">
            [Card 3 title]
          </h3>
          <p style="color: #64748b; font-size: 15px; line-height: 1.6;">
            [Card 3 content]
          </p>
        </div>
        
        <!-- Card 4 - Dark accent -->
        <div style="background: #1e293b; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 24px;">
          <h3 style="font-size: 20px; font-weight: 600; color: white; margin-bottom: 12px;">
            [Card 4 title]
          </h3>
          <p style="color: #cbd5e1; font-size: 15px; line-height: 1.6;">
            [Card 4 content]
          </p>
        </div>
        
        <a href="features.html" style="color: #059669; font-weight: 600; text-decoration: none; margin-top: 8px;">
          See all features →
        </a>
        
      </div>
      
    </div>
    
  </div>
</section>
```

SECTION 4: PROCESS
──────────────────
Current: 3 steps horizontal with icons
Redesign: Diagonal cascade with connectors

Layout:
```html
<section style="padding: 80px 0; background: #f8fafc;">
  <div class="container" style="max-width: 900px;">
    
    <div style="text-align: center; margin-bottom: 60px;">
      <h2 style="font-size: 36px; font-weight: bold; color: #1f2937; margin-bottom: 16px;">
        Live in 10 minutes
      </h2>
      <p style="font-size: 18px; color: #64748b;">
        No API access. No complex setup. Just insights.
      </p>
    </div>
    
    <!-- Diagonal flow -->
    <div style="position: relative;">
      
      <!-- Step 1 -->
      <div style="display: flex; gap: 24px; align-items: start; margin-bottom: 40px;">
        <div style="flex-shrink: 0; width: 80px; height: 80px; background: #059669; color: white; border-radius: 16px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold;">
          <div style="font-size: 32px;">1</div>
          <div style="font-size: 11px; opacity: 0.9;">2 min</div>
        </div>
        <div style="flex: 1; background: white; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
          <h3 style="font-size: 24px; font-weight: 600; color: #1f2937; margin-bottom: 12px;">
            [Step 1 title]
          </h3>
          <p style="color: #64748b; line-height: 1.6;">
            [Step 1 description]
          </p>
        </div>
      </div>
      
      <!-- Connector -->
      <div style="width: 2px; height: 40px; background: #e2e8f0; margin-left: 40px; margin-bottom: 40px;"></div>
      
      <!-- Step 2 - offset right -->
      <div style="display: flex; gap: 24px; align-items: start; margin-bottom: 40px; margin-left: 60px;">
        <div style="flex-shrink: 0; width: 80px; height: 80px; background: #2563eb; color: white; border-radius: 16px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold;">
          <div style="font-size: 32px;">2</div>
          <div style="font-size: 11px; opacity: 0.9;">instant</div>
        </div>
        <div style="flex: 1; background: white; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
          <h3 style="font-size: 24px; font-weight: 600; color: #1f2937; margin-bottom: 12px;">
            [Step 2 title]
          </h3>
          <p style="color: #64748b; line-height: 1.6; margin-bottom: 12px;">
            [Step 2 description]
          </p>
          <span style="display: inline-block; background: #dbeafe; color: #1e40af; padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 600;">
            One-click exports
          </span>
        </div>
      </div>
      
      <!-- Connector -->
      <div style="width: 2px; height: 40px; background: #e2e8f0; margin-left: 100px; margin-bottom: 40px;"></div>
      
      <!-- Step 3 - offset more right -->
      <div style="display: flex; gap: 24px; align-items: start; margin-left: 120px;">
        <div style="flex-shrink: 0; width: 80px; height: 80px; background: #7c3aed; color: white; border-radius: 16px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-weight: bold;">
          <div style="font-size: 32px;">3</div>
          <div style="font-size: 11px; opacity: 0.9;">2 weeks</div>
        </div>
        <div style="flex: 1; background: white; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
          <h3 style="font-size: 24px; font-weight: 600; color: #1f2937; margin-bottom: 12px;">
            [Step 3 title]
          </h3>
          <p style="color: #64748b; line-height: 1.6;">
            [Step 3 description]
          </p>
        </div>
      </div>
      
    </div>
    
    <div style="text-align: center; margin-top: 60px;">
      <a href="how-it-works.html" style="color: #059669; font-weight: 600; text-decoration: none; font-size: 16px;">
        See detailed walkthrough →
      </a>
    </div>
    
  </div>
</section>
```

SECTION 5: TESTIMONIAL
──────────────────────
Current: Simple quote block
Redesign: Dark accent section with quote emphasis

Layout:
```html
<section style="padding: 60px 0; background: linear-gradient(to bottom right, #1e293b, #0f172a); position: relative; overflow: hidden;">
  
  <!-- Large quote mark background -->
  <div style="position: absolute; top: 40px; left: 40px; font-size: 200px; font-family: Georgia, serif; color: rgba(255,255,255,0.05); line-height: 1;">"</div>
  
  <div class="container" style="max-width: 900px; position: relative; z-index: 1;">
    
    <blockquote style="text-align: center; margin-bottom: 40px;">
      <p style="font-size: 32px; font-weight: 600; color: white; line-height: 1.4; margin-bottom: 24px;">
        "[Exact quote text]"
      </p>
      <p style="font-size: 18px; color: #cbd5e1;">
        [Exact subtitle text]
      </p>
    </blockquote>
    
    <div style="text-align: center; margin-top: 40px;">
      <a href="agencies.html" style="color: #34d399; font-weight: 600; text-decoration: none; font-size: 16px;">
        See agency solutions →
      </a>
    </div>
    
  </div>
</section>
```

SECTION 6: FOUNDER STORY
─────────────────────────
Current: Centered link
Redesign: Asymmetric with breathing room

Layout:
```html
<section style="padding: 80px 0; background: white;">
  <div class="container">
    <div style="display: grid; grid-template-columns: 40% 60%; align-items: center;">
      
      <!-- LEFT: Empty for breathing room -->
      <div></div>
      
      <!-- RIGHT: Content -->
      <div>
        <p style="font-size: 18px; color: #64748b; margin-bottom: 16px;">
          [Exact text]
        </p>
        <a href="about.html" style="color: #059669; font-weight: 600; text-decoration: none; font-size: 16px;">
          Read the story →
        </a>
      </div>
      
    </div>
  </div>
</section>
```

SECTION 7: FINAL CTA
────────────────────
Current: Centered CTA
Redesign: Split layout on dark background

Layout:
```html
<section style="padding: 80px 0; background: #0f172a;">
  <div class="container" style="max-width: 1000px;">
    <div style="display: grid; grid-template-columns: 55% 43%; gap: 60px; align-items: center;">
      
      <!-- LEFT: Text -->
      <div>
        <h2 style="font-size: 42px; font-weight: bold; color: white; line-height: 1.2; margin-bottom: 24px;">
          [Exact headline]
        </h2>
        <p style="color: #cbd5e1; font-size: 16px;">
          [Exact subtext]
        </p>
      </div>
      
      <!-- RIGHT: CTA buttons stacked -->
      <div style="display: flex; flex-direction: column; gap: 16px;">
        <a href="audit.html" style="display: block; background: #059669; color: white; padding: 16px 32px; border-radius: 8px; font-weight: 600; text-align: center; text-decoration: none; box-shadow: 0 4px 12px rgba(5,150,105,0.3);">
          [Primary CTA]
        </a>
        <button style="display: block; background: rgba(255,255,255,0.1); color: white; padding: 16px 32px; border-radius: 8px; font-weight: 600; border: 1px solid rgba(255,255,255,0.2); cursor: pointer;">
          [Secondary CTA]
        </button>
      </div>
      
    </div>
  </div>
</section>
```

─────────────────────────────────────────

CRITICAL RULES:
1. Open /desktop/Redesign/index.html and COPY EXACT text (no paraphrasing)
2. Use the layouts I described above (asymmetric, staggered, split)
3. Font sizes: Headlines 42-48px, Metrics 36-42px, Body 16px
4. Colors: #059669 (emerald-600), #1e293b (slate-800), white, #f8fafc (slate-50)
5. NO neon colors, NO glow effects, NO fabrication
6. Every word must match original HTML character-by-character

CONTENT EXTRACTION PROCESS:
1. For each section, open /desktop/Redesign/index.html
2. Find the EXACT text in that section
3. Copy-paste it into your HTML (don't retype, don't paraphrase)
4. Example: If original says "See which Amazon ad decisions actually made money" → Use EXACTLY that
5. If you see [Card 1 title] in my layout → Replace with EXACT title from original HTML

OUTPUT: Create /agent-outputs/homepage-redesigned.html

BEGIN NOW.
```
