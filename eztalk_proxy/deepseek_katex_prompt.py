DEEPSEEK_KATEX_FORMATTING_INSTRUCTION = """**KaTeX & Markdown Formatting Rules**

1.  **Inline Math**: Use `\\( ... \\)` for math within a sentence.
    -   **Correct**: The area is \\(A = \\pi r^2\\).
    -   **Incorrect**: The area is `$A = \\pi r^2$`.

2.  **Display Math**: Use a `math` fenced code block for math on its own line.
    -   **Correct**:
        ```math
        \\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}
        ```
    -   **Incorrect**:
        `$$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$`

3.  **General Markdown**: Do not use KaTeX for text styling like bold or italics. Use standard Markdown (`**bold**`, `*italic*`).

Strictly follow these rules for all mathematical content."""