# Saddle AdPulse Landing Page

Modern, minimal landing page for Saddle AdPulse (Amazon PPC Optimizer) with light beige-grey theme.

**Brand**: Saddle
**Product**: AdPulse

## Brand Colors

The color palette is extracted from the Saddle logo and creates a sophisticated, professional aesthetic:

### Primary Colors
- **Teal Blue**: `#0891B2` - Primary accent, CTAs, interactive elements
- **Dark Navy**: `#2B2D42` - Primary text, headings
- **Lavender**: `#B4B4D1` - Secondary accent (available but subtle)

### Background Colors
- **Primary BG**: `#F5F4F0` - Warm beige-grey main background
- **Secondary BG**: `#EAEAE5` - Slightly darker for section alternation
- **Tertiary BG**: `#FFFFFF` - Pure white for cards and elevated surfaces

### Text Colors
- **Primary Text**: `#2B2D42` - High contrast for body text
- **Secondary Text**: `#5F6368` - Medium contrast for supporting text
- **Muted Text**: `#9CA3AF` - Low contrast for hints and labels

### Shadows
- **Small**: `0 1px 3px rgba(43, 45, 66, 0.08)` - Subtle elevation
- **Medium**: `0 4px 12px rgba(43, 45, 66, 0.1)` - Card hover states
- **Large**: `0 8px 24px rgba(43, 45, 66, 0.12)` - Emphasized elements

## Features

### Design Elements
- Clean, minimal aesthetic with generous white space
- Glassmorphism effects with backdrop blur
- Smooth transitions and hover states
- Gradient accents using brand teal
- Responsive grid layouts
- Professional shadow system

### Key Sections
1. **Hero** - Value proposition with stats
2. **Problem** - Pain points (4-grid)
3. **Solution** - Features (numbered list)
4. **Features Grid** - 6 key capabilities
5. **How It Works** - 3-step process
6. **Social Proof** - Customer testimonials
7. **Pricing** - 3-tier pricing cards
8. **Final CTA** - Conversion-focused
9. **Footer** - Navigation and links

### Interactive Features
- Sticky navbar with scroll effects
- Smooth scroll navigation
- Fade-in animations on scroll
- Button ripple effects
- Card hover transformations

## Customization

### Changing Brand Colors

Edit the CSS variables in `styles.css`:

```css
:root {
    --accent-primary: #0891B2;  /* Change CTA color */
    --text-primary: #2B2D42;     /* Change text color */
    --bg-primary: #F5F4F0;       /* Change background */
}
```

### Adjusting Typography

The site uses **Inter** font family. To change:

1. Update the Google Fonts import in `index.html`
2. Modify the `font-family` in `styles.css`

### Responsive Breakpoints
- Desktop: 1280px max-width
- Tablet: 1024px breakpoint
- Mobile: 768px breakpoint

## Files

- `index.html` - Main HTML structure
- `styles.css` - All styles and brand colors
- `script.js` - Interactive functionality
- `logo.png` - Saddle brand logo
- `README.md` - This file

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

- Optimized CSS with minimal specificity
- Lazy-loading ready (images can be added with loading="lazy")
- Minimal JavaScript for interactions
- No heavy dependencies

## Next Steps

To deploy:
1. Add actual CTA links (currently placeholder "#")
2. Replace demo videos with real product demos
3. Add favicon and meta tags for SEO
4. Implement actual form submission for trial signup
5. Connect to analytics (Google Analytics, etc.)
6. Add GDPR/privacy notice if needed
