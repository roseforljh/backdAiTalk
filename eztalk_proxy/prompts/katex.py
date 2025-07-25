# Enhanced KaTeX and Markdown Formatting Instructions

KATEX_FORMATTING_INSTRUCTION = """Please format all mathematical expressions and equations using proper KaTeX syntax:
- For inline math: use $expression$ (single dollar signs)
- For block math: use $$expression$$ (double dollar signs on separate lines)
- Use proper LaTeX commands: \\frac{numerator}{denominator}, \\sqrt{content}, \\sum_{lower}^{upper}
- Ensure all braces {} are properly matched
- For tables: use proper markdown table format with | separators and header separators
- For code blocks: use ``` with language specification and ensure proper closing"""

DEEPSEEK_KATEX_FORMATTING_INSTRUCTION = """Please format all mathematical expressions and equations using proper KaTeX syntax:
- For inline math: use $expression$ (single dollar signs)
- For block math: use $$expression$$ (double dollar signs on separate lines)
- Use proper LaTeX commands: \\frac{numerator}{denominator}, \\sqrt{content}, \\sum_{lower}^{upper}
- Ensure all braces {} are properly matched
- For tables: use proper markdown table format with | separators and header separators
- For code blocks: use ``` with language specification and ensure proper closing"""

QWEN_KATEX_FORMATTING_INSTRUCTION = """Please format all mathematical expressions and equations using proper KaTeX syntax:
- For inline math: use $expression$ (single dollar signs)
- For block math: use $$expression$$ (double dollar signs on separate lines)
- Use proper LaTeX commands: \\frac{numerator}{denominator}, \\sqrt{content}, \\sum_{lower}^{upper}
- Ensure all braces {} are properly matched
- For tables: use proper markdown table format with | separators and header separators
- For code blocks: use ``` with language specification and ensure proper closing"""

# Enhanced Gemini-specific formatting instruction
GEMINI_ENHANCED_FORMATTING_INSTRUCTION = """IMPORTANT FORMATTING RULES:

1. MATHEMATICAL EXPRESSIONS:
   - Inline math: $expression$ (ensure dollar signs are paired)
   - Block math: $$expression$$ (on separate lines)
   - Fractions: \\frac{numerator}{denominator} (always use braces)
   - Square roots: \\sqrt{content} (always use braces)
   - Summations: \\sum_{lower}^{upper} (use braces for subscripts/superscripts)
   - Integrals: \\int_{lower}^{upper} (use braces for limits)
   - Greek letters: \\alpha, \\beta, \\gamma, etc.

2. TABLE FORMATTING:
   - Start each row with | and end with |
   - Use | to separate columns
   - Include header separator row: |---|---|---|
   - Example:
     | Header1 | Header2 | Header3 |
     |---------|---------|---------|
     | Data1   | Data2   | Data3   |

3. CODE BLOCKS:
   - Start with ```language (specify language)
   - End with ``` (ensure closing)
   - Keep code content intact

4. LISTS:
   - Use consistent markers (- for unordered, 1. for ordered)
   - Maintain proper indentation

5. CRITICAL: Never break mathematical formulas, table rows, or code blocks across multiple output chunks."""