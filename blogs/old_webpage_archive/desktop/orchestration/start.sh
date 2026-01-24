#!/bin/bash

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  SADDLE Mobile App - Agentic Build Setup                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if on feature branch
if command -v git &> /dev/null; then
    BRANCH=$(git branch --show-current 2>/dev/null)
    if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
        echo "⚠️  You're on $BRANCH branch"
        echo ""
        echo "Mobile app should be built on a separate branch for safety."
        echo ""
        read -p "Create feature branch 'feature/mobile-app'? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git checkout -b feature/mobile-app
            echo "✓ Created and switched to: feature/mobile-app"
        fi
    else
        echo "✓ Current branch: $BRANCH"
    fi
else
    echo "⚠️  Git not found (not a git repo?)"
fi

echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""

# Check for PRD
if [ ! -f "prd.md" ]; then
    echo "⚠️  prd.md not found in orchestration/"
    echo ""
    echo "Please create orchestration/prd.md with your PRD content"
    echo "Then run this script again."
    echo ""
    exit 1
fi

echo "✓ PRD found: prd.md"
echo ""

# Run orchestrator
python3 claude_code_orchestrator.py

echo ""
echo "─────────────────────────────────────────────────────────────"
echo ""
echo "NEXT STEP:"
echo ""
echo "1. Open Claude Code in this project"
echo "2. Copy-paste prompts from: claude_code_prompts.txt"
echo "3. Approve 7 phases"
echo "4. Done!"
echo ""
