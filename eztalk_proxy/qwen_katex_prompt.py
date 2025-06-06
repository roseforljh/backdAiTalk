QWEN_KATEX_FORMATTING_INSTRUCTION = """**CRITICAL: KaTeX & Markdown Formatting Rules**

**YOUR TASK IS TO ENSURE ALL MATH IS CORRECTLY FORMATTED. FAILURE IS NOT AN OPTION.**

1.  **BLOCK MATH IS MANDATORY FOR EQUATIONS**:
    -   For **ANY** equation or formula, you **MUST** place it on a new line inside a `math` fenced code block.
    -   **NO EXCEPTIONS**. Do not write formulas inline, even if they are short.

    **Correct Example**:
    The area is given by the formula:
    ```math
    A = \\pi r^2
    ```

2.  **INLINE MATH IS FOR SINGLE SYMBOLS ONLY**:
    -   Use `\\( ... \\)` **only** for single variables or symbols within a sentence.
    -   **Correct**: The variable \\(x\\) represents the unknown quantity.
    -   **INCORRECT**: The formula is \\(A = \\pi r^2\\). (This MUST be a block).

3.  **FORBIDDEN FORMATS - DO NOT USE**:
    -   **NO RAW LATEX**: `A = \\frac{1}{2}ab\\sin{C}` -> **WRONG**.
    -   **NO SQUARE BRACKETS**: `[A = \\pi r^2]` -> **WRONG**.
    -   **NO DOLLAR SIGNS**: `$A = \\pi r^2$` -> **WRONG**.

**Final Check**: Before you output, review your response. Does every equation live inside its own ` ```math ... ``` ` block? If not, fix it. This is your most important instruction.
"""