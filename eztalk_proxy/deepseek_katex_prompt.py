DEEPSEEK_KATEX_FORMATTING_INSTRUCTION = """**Primary Rule: Use plain text whenever possible. Only use LaTeX for complex mathematical expressions that cannot be represented clearly with standard characters.**

**When to use LaTeX:**
- For fractions (e.g., `\\(\\frac{1}{2}\\)`), square roots (e.g., `\\(\\sqrt{x}\\)`), integrals, and complex formulas.
- For special mathematical symbols (e.g., `\\(\\alpha, \\beta, \\sum, \\infty\\)`).

**When NOT to use LaTeX (This is very important):**
- For simple numbers (e.g., "The result is 5.", not "The result is \\(5\\).").
- For simple variables in a sentence (e.g., "Let x be the number.", not "Let \\(x\\) be the number.").
- For basic arithmetic that is clear in text (e.g., "5 * 10 = 50").

**If you MUST use LaTeX, follow these strict formatting rules:**
1.  **Block-level equations (on their own line):** Use `$$...$$`.
2.  **Inline math (within a line of text):** Use `\\(...\\)`.

**Your goal is clarity and minimalism. Avoid unnecessary LaTeX.**
"""