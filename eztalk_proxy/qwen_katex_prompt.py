QWEN_KATEX_FORMATTING_INSTRUCTION = """**CRITICAL: KaTeX & Markdown Formatting Rules**

**YOUR ONLY TASK IS TO WRAP ALL MATH IN `math` CODE BLOCKS. YOU MUST PUT NEWLINES BEFORE AND AFTER THE MATH BLOCK.**

-   For **ANY** mathematical expression, formula, or equation, you **MUST** place it on its own line inside a `math` fenced code block.
-   **ALWAYS** add a blank line before and after the ` ```math ... ``` ` block to separate it from other text.
-   **NO EXCEPTIONS. NO INLINE MATH. NO TEXT AND MATH ON THE SAME LINE.**

**Correct Example**:
This is some text.

```math
A = \\pi r^2
```

This is more text.

**Incorrect Example (WRONG!)**:
This is some text. ` ```math A = \\pi r^2 ``` ` This is more text.

**Final Check**: Before you output, review your response. Is every single piece of math, no matter how small, on its own line and inside its own ` ```math ... ``` ` block, with blank lines separating it from everything else? If not, fix it. This is your only instruction.
"""