# Layout Patterns

## Grid Systems

### Two-Column Split
```css
.grid-2col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
}
```

### Asymmetric Split (60/40)
```css
.grid-60-40 {
  display: grid;
  grid-template-columns: 60fr 40fr;
  gap: 48px;
}
```

### Asymmetric Split (55/45)
```css
.grid-55-45 {
  display: grid;
  grid-template-columns: 55fr 45fr;
  gap: 40px;
}
```

### Three-Column Grid
```css
.grid-3col {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}
```

### Four-Column Grid
```css
.grid-4col {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
}
```

## Section Layouts

### Centered Content
```css
.section-centered {
  max-width: 800px;
  margin: 0 auto;
  text-align: center;
}
```

### Left-Aligned with Visual
```css
.section-split {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 64px;
  align-items: center;
}
```

## Container

```css
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 32px;
}
```

## Responsive Breakpoints

| Name | Value |
|------|-------|
| Mobile | < 768px |
| Tablet | 768px - 1024px |
| Desktop | > 1024px |

### Mobile Adaptations
- All grids collapse to single column
- Section padding reduces to 48px
- Container padding reduces to 16px

## Guardrails

- ❌ NO bento grids
- ❌ NO extreme asymmetry (70/30 or more)
- ❌ NO overlapping/floating elements
- ✅ Simple, predictable layouts
- ✅ Subtle asymmetry only (55/45, 60/40)
- ✅ Generous white space
