# Remotion Explainer Video Implementation Plan

## 1. Project Initialization
- Initialize a new Remotion project within `landing/remotion-video`.
- Template: `blank` (clean slate).
- Tooling: TypeScript, Tailwind CSS.

## 2. Dependencies
- `remotion`: Core library.
- `@remotion/transitions`: For seamless scene chaining (`TransitionSeries`).
- `tailwindcss`: For styling vector graphics and UI mockups.

## 3. Architecture

### File Structure
```
landing/remotion-video/
├── src/
│   ├── components/       # Reusable UI bits (Button, Card, Cursor)
│   ├── scenes/           # The 10 distinct scenes
│   │   ├── Scene1_AmazonSearch.tsx
│   │   ├── Scene2_OperatorOverload.tsx
│   │   ├── Scene3_BrokenChoices.tsx
│   │   ├── Scene4_GoodParams.tsx
│   │   ├── Scene5_AdPulseIntro.tsx
│   │   ├── Scene6_DataIngestion.tsx
│   │   ├── Scene7_Counterfactual.tsx
│   │   ├── Scene8_Confidence.tsx
│   │   ├── Scene9_ScientificMethod.tsx
│   │   └── Scene10_Close.tsx
│   ├── Composition.tsx   # Main timeline assembly
│   ├── index.ts          # Entry point
│   └── constants.ts      # Colors, Fonts, Timing
├── tailwind.config.js
└── package.json
```

### Brand Assets
- **Colors**: Use the Saddl Blue (`#2A8EC9`) as the primary accent.
- **Font**: Load `Inter` from Google Fonts via `@remotion/google-fonts/Inter`.

## 4. Scene Implementation Strategy

We will use **React Components** for each scene. Animations will be driven by `useCurrentFrame` and `spring/interpolate` physics.

**Transitions**:
We will use `<TransitionSeries>` to sequence the 10 scenes with smooth Fades or Slides between them.

**Scene Breakdown**:
1.  **Scene 1 (The Search)**: Vector search bar inputting "running shoes". Results populate (staggered fade-in).
2.  **Scene 2 (Operator Overload)**: Screen fills with generic dashboard widgets (red/green metrics) popping in rapidly to simulate "noise".
3.  **Scene 3 (Broken Choices)**: Split screen. Left: "Set & Forget" (Robot icon). Right: "Manual" (Tired human icon).
4.  **Scene 4 (Good Params)**: UI Card showing "Target ACOS" slider moving.
5.  **Scene 5 (AdPulse Intro)**: Logo bloom animation (Saddl AdPulse).
6.  **Scene 6 (Data Ingestion)**: Particles (data points) flowing from left (Amazon/Google icons) into a central funnel.
7.  **Scene 7 (Counterfactuals)**: Line graph (white line = reality, dotted blue line = counterfactual).
8.  **Scene 8 (Confidence)**: Gauge or Confidence Interval visualization narrowing.
9.  **Scene 9 (Scientific Approach)**: "Hypothesis" -> "Test" -> "Result" flowchart animating.
10. **Scene 10 (Close)**: Final Logo + "Book a Demo" CTA button pulse.

## 5. Execution Steps
1.  **Scaffold**: Run `npx create-remotion@latest` in `landing/`.
2.  **Config**: Setup Tailwind and Fonts.
3.  **Build Components**: Create shared UI components (Search Bar, Graph, Card).
4.  **Animate Scenes**: Implement scenes 1-10 one by one.
5.  **Assemble**: Combine in `Composition.tsx`.
6.  **Render**: Export `explainer.mp4`.
