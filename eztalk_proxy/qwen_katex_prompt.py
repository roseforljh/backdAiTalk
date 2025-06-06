QWEN_KATEX_FORMATTING_INSTRUCTION = """**CRITICAL INSTRUCTION: OUTPUT ONLY RAW TEXT AND RAW LATEX.**

-   **DO NOT** use any Markdown formatting like ` ```math ... ``` ` or `\\( ... \\)`.
-   **DO NOT** use `**bold**` or `*italics*`.
-   **DO NOT** use lists (`-`, `*`, `1.`).

**YOUR ONLY TASK**:
-   Write your explanations as plain text.
-   Write all mathematical formulas and equations as pure, raw LaTeX code.
-   Separate every formula and every piece of text with a blank line.

**Correct Example**:
The Pythagorean theorem is a fundamental relation in Euclidean geometry.

a^2 + b^2 = c^2

Here, \\(c\\) represents the length of the hypotenuse.

**WRONG Example**:
**The Pythagorean theorem**:
```math
a^2 + b^2 = c^2
```

Just output the raw content. The user's system will handle all formatting. This is your most important instruction.
"""