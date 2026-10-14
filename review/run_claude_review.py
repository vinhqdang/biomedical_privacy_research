"""
Automated Peer-Review Orchestrator using Claude Code (Sonnet 5.5)
Invokes the Claude CLI to perform a thorough, rigorous peer review of the Elsevier manuscript.
Captures comprehensive feedback across:
1. Theoretical and mathematical formulation correctness
2. Clinical plausibility and biomedical validity
3. Experimental soundness and statistical reporting
4. Structural organization, language, clarity, and tone
5. LaTeX formatting, equations, tables, figures, and bibliography
"""

import os
import subprocess
import sys

def run_review(round_num=1):
    tex_path = os.path.join("manuscript", "main.tex")
    if not os.path.exists(tex_path):
        print(f"Error: {tex_path} does not exist!")
        return None
        
    with open(tex_path, "r", encoding="utf-8") as f:
        manuscript_content = f.read()
        
    prompt = f"""You are a world-class Senior Editor and Distinguished Peer Reviewer for top-tier biomedical and artificial intelligence journals (such as Elsevier's Artificial Intelligence in Medicine, Nature Medicine, and Lancet Digital Health).

Carefully and rigorously review the following Elsevier LaTeX manuscript titled 'TRACER-Med: A Unified Trustworthy Deep Learning Framework for Differentially Private, Subgroup-Fair, and Uncertainty-Calibrated Clinical Decision Support with Dual-Channel Explainability'.

Your goal is to perform a comprehensive, meticulous audit and identify ALL issues, from foundational conceptual flaws down to the smallest typographical, grammatical, mathematical, or formatting inconsistencies.

Structure your review into the following sections:
1. Overall Assessment & Summary of Contributions
2. Major Strengths
3. Critical Methodological & Mathematical Inquiries / Potential Issues (verify Dirichlet EDL formulas, RDP theorem, DP Conformal Quantiles, SWD equations)
4. Clinical & Biomedical Accuracy (verify clinical realism of EHR variables, ICU acuity metrics, clinical workflow integration)
5. Statistical & Experimental Soundness (evaluation metrics, baseline fairness, ablation insights)
6. Minor Issues, Writing, LaTeX Syntax, and Micro-Edits (line-by-line or section-by-section list of every typo, formatting flaw, notation inconsistency, or awkward phrasing)
7. Final Verdict & Checklist of Actionable Revisions

Here is the complete manuscript LaTeX source:
```latex
{manuscript_content}
```
"""

    out_file = os.path.join("review", f"review_round{round_num}.md")
    print(f"Triggering Claude Code (Sonnet 5.5) for Review Round {round_num}...")
    
    # Run claude -p with prompt piped into stdin
    proc = subprocess.Popen(
        ["claude", "-p"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    
    stdout, stderr = proc.communicate(input=prompt)
    
    if proc.returncode != 0:
        print(f"Error executing claude CLI: {stderr}")
        return None
        
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(stdout)
        
    print(f"Review Round {round_num} successfully generated and saved to {out_file} ({len(stdout)} characters).")
    return stdout

if __name__ == "__main__":
    r_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_review(r_num)
