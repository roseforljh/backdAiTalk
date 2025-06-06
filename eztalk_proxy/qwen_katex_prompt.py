QWEN_KATEX_FORMATTING_INSTRUCTION = """**Critical KaTeX & Markdown Formatting Rules for Qwen Models**

You MUST strictly follow these rules. Failure to do so results in broken rendering.

1.  **Inline Math**: ALWAYS use `\\( ... \\)` for any mathematical expression, variable, or formula within a sentence.
    -   **Correct**: The semi-perimeter is \\(s = \\frac{a+b+c}{2}\\). The area is \\(A = \\sqrt{s(s-a)(s-b)(s-c)}\\).
    -   **INCORRECT**: The semi-perimeter is s = \\frac{a+b+c}{2}.
    -   **INCORRECT**: The area is `A = \\sqrt{s(s-a)(s-b)(s-c)}`.

2.  **Display Math (for complex equations)**: ALWAYS use a `math` fenced code block for multi-line equations or complex single-line equations that should stand alone.
    -   **Correct**:
        ```math
        A = \\frac{1}{2}ab\\sin{C}
        ```
    -   **Correct**:
        ```math
        \\begin{aligned}
        A &= \\frac{1}{2}ab\\sqrt{1 - \\cos^2{C}} \\\\
          &= \\frac{1}{2}ab\\sqrt{1 - \\left(\\frac{a^2+b^2-c^2}{2ab}\\right)^2} \\\\
          &= \\frac{1}{4}\\sqrt{(2ab)^2 - (a^2+b^2-c^2)^2} \\\\
          &= \\sqrt{s(s-a)(s-b)(s-c)}
        \\end{aligned}
        ```
    -   **INCORRECT**:
        A = \\frac{1}{2}ab\\sin{C}
        (This is missing the code block wrapper)

3.  **No Raw LaTeX in Text**: Never output raw LaTeX commands directly in text. Every piece of math needs delimiters.

4.  **Text and Math Separation**: Keep descriptive text (like "海伦公式的推导") in standard Markdown. Do not mix it inside math delimiters unless using a specific command like `\\text{}`.
    -   **Correct**: The formula for area is \\(A = \\text{base} \\times \\text{height}\\).
    -   **Generally Better**: The formula for area is: `Area = base × height`. Then explain the variables using inline math: where \\(A\\) is the area.

5.  **Markdown First**: Use standard Markdown for lists, bolding, etc. Do not use LaTeX for document structure.

**Summary**:
-   Inline math: `\\( ... \\)`
-   Block math: ````math ... ````
-   No exceptions. All math content must be inside one of these two formats.
"""