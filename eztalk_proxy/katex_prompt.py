# file: /app/eztalk_proxy/katex_prompt.py

KATEX_FORMATTING_INSTRUCTION = """\
Your goal is to output Markdown that renders mathematical content correctly using KaTeX.
You MUST strictly follow these rules for ALL mathematical notations.

--- A. GENERAL PRINCIPLES ---
1.  **MATH BLOCKS (Display Math)**: For multi-line equations, complex formulas (e.g., using `\\begin{aligned} ... \\end{aligned}`), or any formula you want displayed on its own line, YOU MUST enclose the entire block within a fenced code block with the `math` language tag:
    \`\`\`math
    \\begin{aligned}
      a &= b + c \\\\
      x &= y - z
    \\end{aligned}
    \`\`\`
    Another example:
    \`\`\`math
    f(x) = \\int_{-\\infty}^\\infty \\hat{f}(\\xi)\\,e^{2 \\pi i \\xi x} \\,d\\xi
    \`\`\`

2.  **INLINE MATH**: For ALL inline mathematical expressions, variables, symbols, and short formulas that appear within a line of text, YOU MUST use `\\(...\\)` delimiters.
    Examples:
    - Variables: `The angle is \\(\\theta\\).`
    - Simple formula: `The projection is \\(l \\cos \\theta\\).`
    - Condition: `This is true if \\(x > 0\\).`
    - Numbers with units in formulas: `The length is \\(7\\text{m}\\).` The angle is \\(73.4^\\circ\\).`
    - Single symbols: `The value of \\(\\pi\\) is approximately \\(3.14\\).`
    - Fractions: `The ratio is \\(\\frac{a}{b}\\).`
    - Comparisons: `If \\(x \\leq y\\), then...`

3.  **TEXT IN MATH**: Use `\\text{...}` for any non-mathematical text (like descriptive words or units in natural language) *inside* a KaTeX math environment (`\\(...\\)` or ` ```math ... ``` `).
    Example: `\\(P_\\text{final} = P_\\text{initial} + \\Delta P_\\text{change}\\)`
    Example with units: `The speed is \\(v = 25 \\text{ m/s}\\).`
    Example with Chinese text: `\\(\\text{水平投影} (l \\cos\\theta)\\)`

4.  **AVOID PLAIN PARENTHESES FOR MATH**: Do NOT use plain parentheses `(...)` or full-width parentheses `（...）` to denote mathematical expressions if KaTeX delimiters are more appropriate. Always use `\\(...\\)`.
    - BAD: `The value is (x^2 + y^2).`
    - GOOD: `The value is \\(x^2 + y^2\\).`
    - BAD: `Condition: (7 \cos\theta \leq 2)`
    - GOOD: `Condition: \\(7 \\cos\\theta \leq 2\\)`

5.  **BARE LATEX IS FORBIDDEN**: Any LaTeX command (e.g., `\\cos`, `\\theta`, `\\frac`, `\\approx`, `\\leq`, `\\geq`, `\\times`, `\\text{m}`, `^\\circ`, `\\alpha`) MUST be within `\\(...\\)` or a ` ```math ... ``` ` block. Do NOT output LaTeX commands directly in the text without these delimiters.
    - BAD: `The angle is θ.`
    - GOOD: `The angle is \\(\\theta\\).`
    - BAD: `cos θ ≤ 2/7`
    - GOOD: `\\(\\cos \\theta \\leq \\frac{2}{7}\\)`
    - BAD: `Result is ≈ 0.2857`
    - GOOD: `Result is \\(\\approx 0.2857\\)`
    - BAD: `The length is l = 7 \text{m}.`
    - GOOD: `The length is \\(l = 7 \\text{m}\\).`

6.  **HANDLING CONTINUOUS/LONG FORMULAS**: When dealing with consecutive mathematical expressions or long/complex formulas:
    a. Wrap each distinct mathematical expression in its own `\\(...\\)` delimiters
    b. Break extremely long formulas into separate KaTeX blocks
    c. For sequences of expressions, clearly separate each with proper delimiters
    d. If a formula becomes too complex for inline rendering (e.g., containing multiple fractions, integrals, or matrices), convert it to a display math block
    e. Ensure all opening and closing delimiters are properly balanced

--- B. COMMON MISTAKES TO AVOID (EXAMPLES) ---
Pay close attention to these specific examples of incorrect and correct formatting:

1.  Incorrect: `要将一根长度 (l = 7 \text{m}) 的甘蔗通过宽度 (w = 2 \text{m}) 的门`
    Correct: `要将一根长度 \\(l = 7 \\text{m}\\) 的甘蔗通过宽度 \\(w = 2 \\text{m}\\) 的门`

2.  Incorrect: `当甘蔗与地面形成夹角 \theta ((0^\circ < \theta \leq 90^\circ))`
    Correct: `当甘蔗与地面形成夹角 \\(\\theta\\) (\\(0^\\circ < \\theta \\leq 90^\\circ\\))`

3.  Incorrect: `水平投影长度为 (l \cos\theta)`
    Correct: `水平投影长度为 \\(l \\cos\\theta\\)`

4.  Incorrect: `通过条件: [l \cos\theta \leq w] 代入数值: [7 \cos\theta \leq 2 implies \cos\theta \leq \frac{2}{7} \approx 0.2857]`
    Correct: `通过条件: \\(l \\cos\\theta \\leq w\\) 代入数值: \\(7 \\cos\\theta \leq 2 \\implies \\cos\\theta \\leq \\frac{2}{7} \\approx 0.2857\\)`

5.  Incorrect: `解不等式: [\theta \geq \cos^{-1}(0.2857) \approx 73.4^\circ]`
    Correct: `解不等式: \\(\\theta \\geq \\cos^{-1}(0.2857) \\approx 73.4^\\circ\\)`

6.  Incorrect: `结论: 倾斜角 \theta \geq 73.4^\circ 时, 水平投影 ( \leq 2\text{m} ), 甘蔗可顺利通过.`
    Correct: `结论: 倾斜角 \\(\\theta \\geq 73.4^\\circ\\) 时, 水平投影 \\( \\leq 2\\text{m} \\), 甘蔗可顺利通过.`

7.  Incorrect: `例如: \theta = 75^\circ 时, 投影长度 ( =7 \cos(75^\circ) \approx 1.81\text{m} < 2\text{m} )`
    Correct: `例如: \\(\\theta = 75^\\circ\\) 时, 投影长度 \\( =7 \\cos(75^\\circ) \\approx 1.81\\text{m} < 2\\text{m} \\)`

8.  Incorrect: `若中途 \downarrow ((\theta < 73.4^\circ)) , 投影会超过 (2\text{m}).`
    Correct: `若中途 \\(\\downarrow\\) ((\\(\\theta < 73.4^\\circ\\))) , 投影会超过 \\(2\\text{m}\\).`

9.  **Specific for `\frac`**:
    Incorrect: `The fraction is \frac{a}{b}.`
    Correct: `The fraction is \\(\\frac{a}{b}\\).`
    Incorrect: `... implies \cos\theta \leq \frac{2}{7} ...`
    Correct: `... \\(\\implies \\cos\\theta \\leq \\frac{2}{7}\\) ...` (assuming the whole expression should be math)

10. **Specific for comparisons followed by variables/numbers**:
    Incorrect: `... if x \leq y then ...`
    Correct: `... if \\(x \\leq y\\) then ...`
    Incorrect: `The condition is \alpha \geq 0.`
    Correct: `The condition is \\(\\alpha \\geq 0\\).`
    Incorrect: `... when \text{value} \approx 5.5 ...`
    Correct: `... when \\(\\text{value} \\approx 5.5\\) ...`

11. **Continuous expressions**:
    Incorrect: `当 \theta \geq \cos^{-1}(0.2857) \approx 73.4^\circ 时成立`
    Correct: `当 \\(\\theta \\geq \\cos^{-1}(0.2857)\\) \\(\\approx 73.4^\\circ\\) 时成立`
    Incorrect: `公式: a=b+c x=y-z`
    Correct: `公式: \\(a = b + c\\), \\(x = y - z\\)`

12. **Long formulas**:
    Incorrect: `结果: f(x) = \int_{-\infty}^{\infty} \frac{\sin(kx)}{x} dx + \sum_{n=1}^{\infty} \frac{(-1)^n}{n^2} \approx 1.64493`
    Correct: 
    ```
    结果:
    \`\`\`math
    f(x) = \int_{-\infty}^{\infty} \frac{\sin(kx)}{x}  dx + \sum_{n=1}^{\infty} \frac{(-1)^n}{n^2} \approx 1.64493
    \`\`\`
    ```
    Or if keeping inline:
    `结果: \\(f(x) = \int_{-\infty}^{\infty} \frac{\sin(kx)}{x}  dx + \sum_{n=1}^{\infty} \frac{(-1)^n}{n^2} \approx 1.64493\\)`

--- C. FINAL CHECK ---
Before outputting, mentally review your response:
1. Does EVERY piece of mathematical notation adhere to rule A2 (inline math with \\(...\\)) or A1 (block math with ```math)? 
2. For consecutive mathematical expressions, is each wrapped in its own delimiters?
3. Are long/complex formulas properly broken into display math blocks?
4. Are all delimiters balanced (every \$ has a matching \$)?
5. Is there any LaTeX command outside math environments?
6. Are units and descriptive text inside math environments wrapped in \\text{}?

There should be NO exceptions. Pay special attention to continuous expressions and long formulas to prevent rendering errors.
"""
