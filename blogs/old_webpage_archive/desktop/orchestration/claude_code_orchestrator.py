#!/usr/bin/env python3
"""
Claude Code Agentic Orchestrator
Runs inside project directory with full codebase access
"""

import os
import json
from pathlib import Path
from datetime import datetime

class ClaudeCodeOrchestrator:
    """
    Orchestrates mobile app build with direct access to desktop codebase
    """
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.desktop_path = self.project_root / "desktop"
        self.mobile_path = self.project_root / "mobile"
        self.orchestration_path = self.project_root / "orchestration"
        
        # Handle if PRD doesn't exist yet
        prd_file = self.orchestration_path / "prd.md"
        if prd_file.exists():
            self.prd = prd_file.read_text()
        else:
            print(f"⚠️  PRD not found at {prd_file}")
            print("Please create orchestration/prd.md before continuing")
            self.prd = "[PRD TO BE ADDED]"
        
        # Verify desktop path exists
        if not self.desktop_path.exists():
            print(f"⚠️  Desktop app not found at {self.desktop_path}")
            print("This orchestrator expects your desktop app to be in ./desktop/")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                raise Exception("Desktop app directory not found")
        
    def get_context_aware_system_prompt(self):
        """System prompt with instructions to read desktop codebase"""
        return f"""You are Master Agent orchestrating a mobile app build using Claude Code.

PROJECT STRUCTURE:
- Desktop app: {self.desktop_path}
- Mobile app: {self.mobile_path} (you will create this)
- You have FULL FILE SYSTEM ACCESS to read desktop code

CRITICAL INSTRUCTION - CONTEXT AWARENESS:
Before building any component, you MUST:
1. Read relevant desktop code for patterns
2. Extract colors, fonts, spacing from desktop stylesheets
3. Review desktop component structure
4. Check desktop API integration patterns
5. Match desktop's visual language in mobile

DESKTOP CODEBASE REFERENCE:
You can read any file in {self.desktop_path}:
- Read {self.desktop_path}/src/styles/* for colors, fonts, spacing
- Read {self.desktop_path}/src/components/* for component patterns
- Read {self.desktop_path}/src/api/* for API endpoint definitions
- Read {self.desktop_path}/src/types/* for data models

MOBILE APP LOCATION:
Create all mobile app files in: {self.mobile_path}/
Keep mobile app completely separate from desktop.

PRD (Build Specifications):
{self.prd}

WORKFLOW FOR EACH PHASE:
1. [CONTEXT GATHERING] Read relevant desktop files
2. [AGENT WORK] Build mobile component matching desktop patterns
3. [REVIEW] Verify consistency with desktop
4. [PRESENT] Show summary for approval

OUTPUT FORMAT:
[PHASE X] [Name]

[CONTEXT GATHERING]
Reading: desktop/src/styles/colors.css
Extracted: Primary color #10B981, Font: Inter
Reading: desktop/src/components/HealthScore.tsx
Extracted: Score displays as circle with trend indicator

[AGENT A] Working...
Creating: mobile/src/components/HealthScore.tsx
[outputs code that matches desktop pattern]

[REVIEW]
✅ Colors match desktop: #10B981
✅ Font matches desktop: Inter
✅ Component structure consistent
✅ All PRD requirements met

[APPROVAL REQUIRED]

EXECUTION RULES:
1. Always read desktop code before building
2. Match desktop's visual language (colors, fonts, spacing)
3. Adapt layout for mobile, preserve visual identity
4. Create mobile files in {self.mobile_path}
5. NEVER modify files in {self.desktop_path}

BEGIN: Start with Phase 1 - Foundation
"""
    
    def create_mobile_directory(self):
        """Initialize mobile app directory structure"""
        if self.mobile_path.exists():
            print(f"⚠️  {self.mobile_path} already exists")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        self.mobile_path.mkdir(exist_ok=True)
        print(f"✓ Mobile directory ready: {self.mobile_path}")
        return True
    
    def get_startup_prompt(self):
        """Initial prompt to start orchestration"""
        return f"""Initialize mobile app build:

1. Verify you can access desktop codebase at: {self.desktop_path}
2. List key directories in desktop app
3. Begin Phase 1 - Foundation

Start by reading desktop code to understand:
- What styling system is used? (CSS? Styled-components? Tailwind?)
- What component library? (Custom? Material-UI? Other?)
- What state management? (Redux? Context? Other?)
- What API client? (Fetch? Axios? Other?)

Then initialize Expo React Native project in {self.mobile_path}/

Begin now.
"""
    
    def generate_desktop_context_summary(self):
        """Generate summary of desktop codebase structure"""
        summary = ["DESKTOP CODEBASE STRUCTURE:\n"]
        
        if not self.desktop_path.exists():
            summary.append("Desktop directory not found - will create mobile app from PRD only")
            return "\n".join(summary)
        
        # Common directories to check
        key_dirs = [
            "src/styles",
            "src/components", 
            "src/api",
            "src/types",
            "src/utils",
            "src/hooks"
        ]
        
        for dir_name in key_dirs:
            dir_path = self.desktop_path / dir_name
            if dir_path.exists():
                files = [f.name for f in dir_path.iterdir() if f.is_file()][:10]
                summary.append(f"\n{dir_name}/")
                summary.append(f"  Files: {', '.join(files)}")
                if len(list(dir_path.iterdir())) > 10:
                    summary.append(f"  ... and {len(list(dir_path.iterdir())) - 10} more")
        
        return "\n".join(summary)
    
    def verify_git_branch(self):
        """Ensure we're on feature branch, not main"""
        import subprocess
        
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            current_branch = result.stdout.strip()
            
            if current_branch == "main" or current_branch == "master":
                print("⚠️  WARNING: You're on the main branch!")
                print("Mobile app should be built on a separate branch.")
                print("\nRecommended:")
                print("  git checkout -b feature/mobile-app")
                response = input("\nContinue anyway? (not recommended) (y/n): ")
                if response.lower() != 'y':
                    return False
            else:
                print(f"✓ Current branch: {current_branch}")
        except:
            print("⚠️  Could not detect git branch (not a git repo?)")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        return True
    
    def run(self):
        """Main execution"""
        print("="*60)
        print("CLAUDE CODE AGENTIC ORCHESTRATOR")
        print("="*60)
        print(f"Project root: {self.project_root}")
        print(f"Desktop app: {self.desktop_path}")
        print(f"Mobile app: {self.mobile_path}")
        print()
        
        # Verify git branch
        if not self.verify_git_branch():
            print("Exiting. Please switch to feature branch first.")
            return
        
        # Create mobile directory
        if not self.create_mobile_directory():
            print("Exiting.")
            return
        
        print("\n" + "="*60)
        print("DESKTOP CODEBASE ANALYSIS")
        print("="*60)
        print(self.generate_desktop_context_summary())
        
        print("\n" + "="*60)
        print("CLAUDE CODE INSTRUCTIONS")
        print("="*60)
        print("""
1. Open Claude Code in this project directory:
   
   In terminal (from project root):
   $ code .
   
   Or if Claude Code CLI installed:
   $ claude-code

2. In Claude Code, paste this initial prompt:
""")
        
        # Save prompts to files for easy copy-paste
        prompts_file = self.orchestration_path / "claude_code_prompts.txt"
        with open(prompts_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SYSTEM PROMPT (paste first)\n")
            f.write("="*60 + "\n\n")
            f.write(self.get_context_aware_system_prompt())
            f.write("\n\n" + "="*60 + "\n")
            f.write("STARTUP PROMPT (paste second)\n")
            f.write("="*60 + "\n\n")
            f.write(self.get_startup_prompt())
        
        print(f"\n✓ Prompts saved to: {prompts_file}")
        print("\nOpen that file and copy-paste into Claude Code.")
        
        print("\n" + "="*60)
        print("WHAT HAPPENS NEXT")
        print("="*60)
        print("""
Claude Code will:
1. Read your desktop codebase
2. Extract design patterns automatically
3. Build mobile app matching desktop
4. Request approval after each phase (7 phases total)

You:
1. Type "approve" after each phase
2. Total time: ~70 seconds of your time
3. Agent time: ~7 hours (runs automatically)

Result:
- Mobile app in ./mobile/
- Matches desktop visually
- Works with your APIs
- Completely isolated from desktop code
        """)

def main():
    import sys
    
    # Auto-detect project root (assumes we're in orchestration/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"Detected project root: {project_root}")
    print()
    
    orchestrator = ClaudeCodeOrchestrator(project_root)
    orchestrator.run()

if __name__ == "__main__":
    main()
