QWEN_KATEX_FORMATTING_INSTRUCTION = """**CRITICAL RULE: You MUST wrap all mathematical content in LaTeX delimiters.**

- **Block-level equations**: Use `$$...$$`.
- **Inline equations and single variables**: Use `\\(...\\)`.

**Failure to do so will break the user's display.**

**Example:**
The formula for the area of a circle is \\(A = \\pi r^2\\).
A more complex formula is:
$$
x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}
$$
This rule is mandatory. Do not use any other markdown for math.
"""