#!/usr/bin/env python3
"""
Claude Code Agentic Orchestrator - Fixed for Real Project Structure
Works when desktop app IS the project root, not a subdirectory
"""

import os
import json
from pathlib import Path
from datetime import datetime

class ClaudeCodeOrchestrator:
    """
    Orchestrates mobile app build with direct access to desktop codebase
    Handles real-world scattered file structures
    """
    
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        
        # Desktop app IS the current directory (not a subdirectory)
        self.desktop_path = self.project_root
        
        # Mobile app will be a sibling directory
        self.mobile_path = self.project_root.parent / "mobile"
        
        # Orchestration files in subdirectory
        self.orchestration_path = self.project_root / "orchestration"
        
        # Handle if PRD doesn't exist yet
        prd_file = self.orchestration_path / "prd.md"
        if prd_file.exists():
            self.prd = prd_file.read_text()
        else:
            print(f"⚠️  PRD not found at {prd_file}")
            print("Please create orchestration/prd.md before continuing")
            self.prd = "[PRD TO BE ADDED]"
        
        print(f"✓ Desktop app detected at: {self.desktop_path}")
        print(f"✓ Mobile app will be created at: {self.mobile_path}")
    
    def analyze_codebase_structure(self):
        """
        Analyze actual project structure to understand where files are
        """
        print("\nAnalyzing codebase structure...")
        
        structure = {
            "root_files": [],
            "directories": {},
            "key_files": {
                "styles": [],
                "components": [],
                "api": [],
                "types": [],
                "config": []
            }
        }
        
        # Get root level files
        for item in self.desktop_path.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                structure["root_files"].append(item.name)
        
        # Common patterns to look for
        search_patterns = {
            "styles": ["*.css", "*.scss", "*.less", "*style*", "*theme*"],
            "components": ["*component*", "*Component*"],
            "api": ["*api*", "*Api*", "*service*", "*Service*"],
            "types": ["*type*", "*Type*", "*.d.ts"],
            "config": ["*config*", "*Config*", "package.json", "tsconfig.json"]
        }
        
        # Walk through directories (limited depth to avoid huge scan)
        for root, dirs, files in os.walk(self.desktop_path):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if d not in [
                'node_modules', '.git', 'dist', 'build', '__pycache__',
                '.next', '.nuxt', 'coverage', '.cache'
            ]]
            
            rel_path = Path(root).relative_to(self.desktop_path)
            
            # Stop at 4 levels deep
            if len(rel_path.parts) > 4:
                continue
            
            # Categorize files
            for file in files:
                file_lower = file.lower()
                file_path = Path(root) / file
                rel_file_path = file_path.relative_to(self.desktop_path)
                
                # Check against patterns
                for category, patterns in search_patterns.items():
                    for pattern in patterns:
                        if pattern.replace("*", "") in file_lower:
                            structure["key_files"][category].append(str(rel_file_path))
                            break
        
        return structure
    
    def generate_codebase_map(self, structure):
        """Generate human-readable codebase map"""
        lines = []
        lines.append("\n" + "="*60)
        lines.append("CODEBASE MAP")
        lines.append("="*60)
        
        lines.append("\nRoot Level Files:")
        for f in structure["root_files"][:20]:
            lines.append(f"  - {f}")
        if len(structure["root_files"]) > 20:
            lines.append(f"  ... and {len(structure["root_files"]) - 20} more")
        
        lines.append("\nKey Files by Category:")
        for category, files in structure["key_files"].items():
            if files:
                lines.append(f"\n{category.upper()}:")
                for f in files[:10]:
                    lines.append(f"  - {f}")
                if len(files) > 10:
                    lines.append(f"  ... and {len(files) - 10} more")
        
        return "\n".join(lines)
    
    def get_context_aware_system_prompt(self, structure):
        """
        System prompt that tells agents where to find things in the actual codebase
        """
        
        # Build file location hints
        file_hints = []
        if structure["key_files"]["styles"]:
            file_hints.append(f"Styles/Colors: {', '.join(structure['key_files']['styles'][:3])}")
        if structure["key_files"]["components"]:
            file_hints.append(f"Components: {', '.join(structure['key_files']['components'][:3])}")
        if structure["key_files"]["api"]:
            file_hints.append(f"API/Services: {', '.join(structure['key_files']['api'][:3])}")
        if structure["key_files"]["types"]:
            file_hints.append(f"Types: {', '.join(structure['key_files']['types'][:3])}")
        
        file_hints_str = "\n".join(f"- {hint}" for hint in file_hints)
        
        return f"""You are Master Agent orchestrating a mobile app build using Claude Code.

PROJECT STRUCTURE:
Desktop app (current codebase): {self.desktop_path}
Mobile app (will create): {self.mobile_path}

IMPORTANT: The desktop codebase is scattered across many files and directories.
You have FULL FILE SYSTEM ACCESS to explore and read any file.

DESKTOP CODEBASE STRUCTURE:
{file_hints_str}

HOW TO NAVIGATE THE CODEBASE:
1. Use `ls` command to explore directories
2. Use `find` command to locate specific files
3. Use `grep` command to search for patterns
4. Read any file to understand patterns

Examples:
- Find all CSS files: `find {self.desktop_path} -name "*.css" -type f`
- Find components: `find {self.desktop_path} -name "*component*" -type f`
- Search for colors: `grep -r "color" {self.desktop_path}/src --include="*.css"`
- Search for API calls: `grep -r "fetch\\|axios" {self.desktop_path}/src --include="*.ts"`

CRITICAL INSTRUCTION - CONTEXT AWARENESS:
Before building ANY mobile component:

1. EXPLORE: Use ls/find to locate relevant desktop files
2. READ: Read those files to extract patterns
3. EXTRACT: Colors, fonts, spacing, API endpoints, component structure
4. MATCH: Build mobile version matching desktop patterns
5. VERIFY: Confirm visual and functional consistency

MOBILE APP LOCATION:
Create all mobile app files in: {self.mobile_path}/
This is a SIBLING directory to desktop app (completely separate)

PRD (Build Specifications):
{self.prd}

WORKFLOW FOR EACH PHASE:
1. [EXPLORATION] Explore desktop codebase to find relevant files
2. [CONTEXT GATHERING] Read files and extract patterns
3. [AGENT WORK] Build mobile component matching desktop patterns
4. [REVIEW] Verify consistency with desktop
5. [PRESENT] Show summary for approval

OUTPUT FORMAT:
[PHASE X] [Name]

[EXPLORATION]
$ find {self.desktop_path} -name "*color*"
Found: src/styles/colors.css, src/theme/palette.ts

[CONTEXT GATHERING]
Reading: src/styles/colors.css
Extracted: Primary #10B981, Error #EF4444
Reading: src/theme/palette.ts  
Extracted: Font family Inter, base spacing 8px

[AGENT A] Working...
Creating: {self.mobile_path}/src/theme/colors.ts
```typescript
// Extracted from desktop: src/styles/colors.css
export const colors = {{
  primary: '#10B981', // matches desktop
  error: '#EF4444',   // matches desktop
  ...
}}
```

[REVIEW]
✅ Colors match desktop: src/styles/colors.css
✅ All PRD requirements met

[APPROVAL REQUIRED]

EXECUTION RULES:
1. Always explore/find files before assuming structure
2. Read actual files, don't guess
3. Match desktop's exact colors, fonts, spacing
4. Create mobile files in {self.mobile_path}/ only
5. NEVER modify files in {self.desktop_path}/
6. Use bash commands (ls, find, grep, cat) liberally to explore

BEGIN: Start with Phase 1 - Foundation
"""
    
    def create_mobile_directory(self):
        """Initialize mobile app directory structure"""
        if self.mobile_path.exists():
            print(f"\n⚠️  {self.mobile_path} already exists")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        self.mobile_path.mkdir(exist_ok=True)
        print(f"✓ Mobile directory ready: {self.mobile_path}")
        return True
    
    def get_startup_prompt(self):
        """Initial prompt to start orchestration"""
        return f"""Initialize mobile app build:

Phase 1 - Foundation & Exploration

STEP 1: Explore desktop codebase
Run these commands to understand structure:

```bash
# Find package.json to understand tech stack
cat {self.desktop_path}/package.json | grep -A 20 '"dependencies"'

# Find styling files
find {self.desktop_path} -name "*.css" -o -name "*.scss" -o -name "*style*" | head -20

# Find component files
find {self.desktop_path} -name "*component*" -o -name "*Component*" | head -20

# Find API/service files  
find {self.desktop_path} -name "*api*" -o -name "*service*" | head -20
```

STEP 2: Extract key patterns
Based on what you find:
1. What styling system? (CSS, Tailwind, styled-components, etc.)
2. What component framework? (React, Vue, etc.)
3. What API client? (fetch, axios, etc.)
4. What colors are used?
5. What fonts are used?

STEP 3: Initialize mobile app
Create Expo React Native project in {self.mobile_path}/

STEP 4: Present findings and request approval

Begin exploration now.
"""
    
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
                print("\n⚠️  WARNING: You're on the main branch!")
                print("Mobile app will be created as sibling directory, but you should")
                print("still use a feature branch for tracking changes.")
                print("\nRecommended:")
                print("  git checkout -b feature/mobile-app")
                response = input("\nContinue anyway? (not recommended) (y/n): ")
                if response.lower() != 'y':
                    return False
            else:
                print(f"✓ Current branch: {current_branch}")
        except:
            print("⚠️  Could not detect git branch")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        return True
    
    def run(self):
        """Main execution"""
        print("\n" + "="*60)
        print("CLAUDE CODE AGENTIC ORCHESTRATOR")
        print("="*60)
        print(f"\nProject structure:")
        print(f"  Desktop: {self.desktop_path}")
        print(f"  Mobile:  {self.mobile_path}")
        print(f"  PRD:     {self.orchestration_path / 'prd.md'}")
        
        # Verify git branch
        if not self.verify_git_branch():
            print("\nExiting. Please switch to feature branch first.")
            return
        
        # Analyze codebase
        print("\nThis may take a moment...")
        structure = self.analyze_codebase_structure()
        print(self.generate_codebase_map(structure))
        
        # Create mobile directory
        if not self.create_mobile_directory():
            print("\nExiting.")
            return
        
        print("\n" + "="*60)
        print("CLAUDE CODE SETUP")
        print("="*60)
        
        # Save prompts to files for easy copy-paste
        prompts_file = self.orchestration_path / "claude_code_prompts.txt"
        with open(prompts_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SYSTEM PROMPT (paste this first in Claude Code)\n")
            f.write("="*60 + "\n\n")
            f.write(self.get_context_aware_system_prompt(structure))
            f.write("\n\n" + "="*60 + "\n")
            f.write("STARTUP PROMPT (paste this second in Claude Code)\n")
            f.write("="*60 + "\n\n")
            f.write(self.get_startup_prompt())
        
        print(f"\n✓ Prompts saved to: {prompts_file}")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print(f"""
1. Open Claude Code in desktop directory:
   $ cd {self.desktop_path}
   $ code .

2. Open: {prompts_file}

3. Copy System Prompt → Paste in Claude Code

4. Copy Startup Prompt → Paste in Claude Code

5. Claude Code will:
   - Explore your codebase
   - Extract patterns automatically  
   - Build mobile app matching desktop
   - Request approval 7 times

6. You type "approve" after each phase

Result:
- Mobile app in: {self.mobile_path}
- Matches desktop visually
- Zero changes to desktop code
        """)

def main():
    import sys
    
    # Project root is where this script is located (in orchestration/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"Detected project root: {project_root}")
    
    orchestrator = ClaudeCodeOrchestrator(project_root)
    orchestrator.run()

if __name__ == "__main__":
    main()
