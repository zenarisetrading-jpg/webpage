# Component Library

## Buttons

### Primary Button
```css
.btn-primary {
  background: #059669;
  color: white;
  padding: 14px 28px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 16px;
  border: none;
  transition: background 0.2s ease;
}
.btn-primary:hover {
  background: #047857;
}
```

### Secondary Button
```css
.btn-secondary {
  background: transparent;
  color: #0f172a;
  padding: 14px 28px;
  border-radius: 8px;
  font-weight: 600;
  border: 1px solid #e2e8f0;
  transition: border-color 0.2s ease;
}
.btn-secondary:hover {
  border-color: #94a3b8;
}
```

### Text Link
```css
.text-link {
  color: #059669;
  font-weight: 600;
  text-decoration: none;
}
.text-link:hover {
  text-decoration: underline;
}
```

## Cards

### Standard Card
```css
.card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 40px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
```

### Dark Card
```css
.card-dark {
  background: #1e293b;
  color: white;
  border: none;
  border-radius: 12px;
  padding: 40px;
}
```

### Metric Card
```css
.card-metric {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 32px;
  text-align: center;
}
```

## Badges

### Status Badge
```css
.badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 100px;
  font-size: 12px;
  font-weight: 600;
}
.badge-success {
  background: rgba(5, 150, 105, 0.1);
  color: #059669;
}
.badge-danger {
  background: rgba(220, 38, 38, 0.1);
  color: #dc2626;
}
```

## Icons

- Size: 20-24px for inline, 40-48px for feature icons
- Stroke width: 1.5-2px
- Color: Inherit from parent or use Slate-600

## Guardrails

- ❌ NO glow effects
- ❌ NO neon colors
- ❌ NO gradient backgrounds on cards
- ❌ NO heavy shadows
- ✅ Subtle, professional styling
- ✅ Minimal hover states
