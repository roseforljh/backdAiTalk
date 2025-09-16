import json_repair
import re
import logging
from typing import Dict, Any, Optional, List
import json
from .format_config import get_format_config, FormatRepairConfig

logger = logging.getLogger("EzTalkProxy.Services.FormatRepair")

class AIOutputFormatRepair:
    """
    AI输出格式修复服务
    用于修复和规范化AI的输出格式，确保输出符合预期格式
    """
    
    def __init__(self):
        self.logger = logger
        self.config = get_format_config()
        
        # KaTeX接管后，后端不再需要复杂的数学修复，只需保证基本格式
    
    def repair_ai_output(self, text: str, output_type: str = "general") -> str:
        """
        修复AI输出格式 - 安全增强版本
        
        Args:
            text: 原始AI输出文本
            output_type: 输出类型 ("general", "math", "code", "json")
            
        Returns:
            修复后的文本
        """
        # 安全检查：空文本直接返回
        if not text or not text.strip():
            return text
            
        # 检查是否启用格式修复
        if not self.config.enable_format_repair:
            self.logger.debug("Format repair is disabled, returning original text")
            return text
            
        # 检查文本长度限制
        if len(text) > self.config.max_text_length:
            self.logger.warning(f"Text length {len(text)} exceeds limit {self.config.max_text_length}")
            return text
        
        # 安全检查：避免修复正常的完整内容
        if self._is_safe_content_that_should_not_be_modified(text):
            self.logger.debug("Content appears safe and complete, skipping repair")
            return text
            
        try:
            original_text = text
            repaired_text = text
            
            # 记录修复前的状态
            original_length = len(text)
            original_lines = text.count('\n')
            
            # 1. 极度保守的基础格式清理
            repaired_text = self._safe_basic_format_cleanup(repaired_text)
            
            # 安全检查：确保基础清理没有删除重要内容
            if len(repaired_text.strip()) < len(original_text.strip()) * 0.7:
                self.logger.warning("Basic cleanup removed too much content, reverting")
                repaired_text = original_text
            
            # 2. 根据输出类型进行专门修复（仅在安全时进行）
            if output_type == "json" and self.config.enable_json_repair:
                repaired_text = self._safe_repair_json_format(repaired_text, original_text)
            elif output_type == "math" and self.config.enable_math_repair:
                repaired_text = self._safe_repair_math_format(repaired_text, original_text)
            elif output_type == "code" and self.config.enable_code_repair:
                repaired_text = self._safe_repair_code_format(repaired_text, original_text)
            else:
                # 通用修复（极度保守）
                repaired_text = self._safe_repair_general_format(repaired_text, original_text)
            
            # 3. 最终安全检查和清理
            repaired_text = self._safe_final_cleanup(repaired_text, original_text)
            
            # 最终安全验证：确保修复后的内容仍然有意义
            if not self._validate_repaired_content(repaired_text, original_text):
                self.logger.warning("Repaired content failed validation, returning original")
                return original_text
            
            # 记录修复统计
            final_length = len(repaired_text)
            final_lines = repaired_text.count('\n')
            self.logger.debug(f"Repair stats: {original_length}->{final_length} chars, {original_lines}->{final_lines} lines")
            
            return repaired_text
            
        except Exception as e:
            self.logger.error(f"Error repairing AI output: {e}")
            return text  # 出错时返回原始文本
    
    def _is_safe_content_that_should_not_be_modified(self, text: str) -> bool:
        """
        检查内容是否是应该保持原样的安全内容
        只有非常特殊的情况才跳过修复
        """
        if not text or len(text.strip()) < 5:
            return True
            
        stripped = text.strip()
        
        # 只有以下情况才跳过修复：
        # 1. 已经是完美格式的JSON
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                json.loads(stripped)
                return True  # 有效的JSON，不需要修复
            except:
                pass
        
        # 2. 已经包含正确格式的数学公式（有$符号包围）
        if '$' in stripped and stripped.count('$') >= 2:
            # 检查是否已经有正确的数学公式格式
            math_pattern = r'\$[^$]+\$|\$\$[^$]+\$\$'
            if re.search(math_pattern, stripped):
                # 即使有数学公式，也允许修复，因为可能有其他需要修复的内容
                return False
        
        # 3. 纯代码块内容（完全被```包围）
        if stripped.startswith('```') and stripped.endswith('```'):
            return True
            
        # 其他所有情况都允许修复
        return False
    
    def _safe_basic_format_cleanup(self, text: str) -> str:
        """极度保守的基础格式清理"""
        if not text:
            return text
            
        original_text = text
        
        # 只进行最基本和最安全的清理
        # 1. 仅移除过多的连续空行（超过3个）
        cleaned = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # 2. 仅移除行尾的制表符和空格（但保留内容）
        cleaned = re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)
        
        # 安全检查：如果清理导致内容显著减少，恢复原文
        if len(cleaned.strip()) < len(original_text.strip()) * 0.95:
            return original_text
            
        return cleaned
    
    def _basic_format_cleanup(self, text: str) -> str:
        """基础格式清理 - 已被安全版本替代"""
        return self._safe_basic_format_cleanup(text)
    
    def _repair_json_format(self, text: str) -> str:
        """修复JSON格式"""
        if not self.config.enable_json_repair:
            return text
            
        try:
            # 尝试从 ```json ... ``` 代码块中提取内容
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            json_to_repair = match.group(1) if match else text

            # LaTeX 关键词，如果文本中包含这些，则很可能不是 JSON
            latex_keywords = ['\\frac', '\\sum', '\\int', '\\lim', '\\sqrt']
            if any(keyword in json_to_repair for keyword in latex_keywords):
                self.logger.warning(f"Skipping JSON repair for suspected LaTeX content: {json_to_repair[:70]}...")
                return text

            try:
                # 直接对提取出的或原始的文本进行修复
                repaired_json = json_repair.repair_json(json_to_repair)
                # 验证修复后的JSON是否有效
                json.loads(repaired_json)
                
                # 如果原始文本包含代码块，则替换代码块内部
                if match:
                    # 使用 re.sub 进行更安全的替换
                    text = re.sub(re.escape(match.group(1)), repaired_json, text, 1)
                else:
                    text = repaired_json
                
                self.logger.info("Successfully repaired JSON format")
            except Exception as e:
                self.logger.warning(f"Could not repair JSON content for text: '{text[:50]}...'. Error: {e}")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error in _repair_json_format: {e}")
            return text
    
    def _safe_repair_json_format(self, text: str, original_text: str) -> str:
        """安全的JSON修复"""
        if not self.config.enable_json_repair:
            return text
            
        # 检查是否真的是JSON内容
        stripped = text.strip()
        if not (stripped.startswith('{') or stripped.startswith('[')):
            return text
            
        try:
            # 只尝试修复明显的JSON结构
            return self._repair_json_format(text)
        except Exception as e:
            self.logger.warning(f"Safe JSON repair failed: {e}")
            return original_text
    
    def _safe_repair_math_format(self, text: str, original_text: str) -> str:
        """安全的数学公式修复"""
        if not self.config.enable_math_repair:
            return text
            
        # 只修复明显包含数学公式的内容
        if not any(marker in text for marker in ['^', '_', '\\frac', '\\sqrt', '=']):
            return text
            
        try:
            return self._repair_math_format(text)
        except Exception as e:
            self.logger.warning(f"Safe math repair failed: {e}")
            return original_text
    
    def _safe_repair_code_format(self, text: str, original_text: str) -> str:
        """安全的代码修复"""
        if not self.config.enable_code_repair:
            return text
            
        # 只修复明显包含代码的内容
        if '```' not in text and '`' not in text:
            return text
            
        try:
            return self._repair_code_format(text)
        except Exception as e:
            self.logger.warning(f"Safe code repair failed: {e}")
            return original_text
    
    def _safe_repair_general_format(self, text: str, original_text: str) -> str:
        """安全的通用格式修复"""
        # 极度保守的通用修复，主要针对明显的格式问题
        try:
            repaired = text
            
            # 只修复明显的Markdown格式问题
            if self.config.enable_markdown_repair:
                # 修复明显缺少空格的标题
                if re.search(r'^#{1,6}[^#\s]', repaired, re.MULTILINE):
                    repaired = re.sub(r'^(#{1,6})([^#\s])', r'\1 \2', repaired, flags=re.MULTILINE)
            
            return repaired
        except Exception as e:
            self.logger.warning(f"Safe general repair failed: {e}")
            return original_text
    
    def _safe_final_cleanup(self, text: str, original_text: str) -> str:
        """安全的最终清理"""
        try:
            # 只进行最基本的最终清理
            cleaned = text.strip()
            
            # 安全检查
            if len(cleaned) < len(original_text.strip()) * 0.8:
                return original_text
                
            return cleaned
        except Exception as e:
            self.logger.warning(f"Safe final cleanup failed: {e}")
            return original_text
    
    def _validate_repaired_content(self, repaired: str, original: str) -> bool:
        """验证修复后的内容是否合理"""
        if not repaired or not repaired.strip():
            return False
            
        # 长度检查：修复后的内容不应比原始内容短太多
        if len(repaired.strip()) < len(original.strip()) * 0.5:
            return False
            
        # 内容完整性检查：重要字符不应丢失
        important_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\u4e00-\u9fa5')
        original_important = set(c for c in original if c in important_chars)
        repaired_important = set(c for c in repaired if c in important_chars)
        
        # 如果重要字符丢失超过20%，认为修复失败
        if len(repaired_important) < len(original_important) * 0.8:
            return False
            
        return True

    def _repair_math_format(self, text: str) -> str:
        """
        增强数学公式修复 - 自动包装数学表达式
        """
        if not self.config.enable_math_repair:
            return text
            
        try:
            repaired = text
            
            # 1. 确保块级公式前后有换行
            repaired = re.sub(r'([^\n])\$\$([^$]+)\$\$', r'\1\n$$\2$$', repaired)
            repaired = re.sub(r'\$\$([^$]+)\$\$([^\n])', r'$$\1$$\n\2', repaired)
            
            # 2. 自动包装数学表达式（如果启用 math_auto_wrap）
            if self.config.math_auto_wrap:
                # 检测常见的数学表达式模式并自动添加 $ 符号
                
                # 匹配类似 "E = mc^2" 的简单方程
                simple_equation_pattern = r'\b([A-Za-z]+\s*=\s*[A-Za-z0-9\^{}\\+\-\*/\(\)\s]+)'
                repaired = re.sub(simple_equation_pattern, r'$\1$', repaired)
                
                # 匹配省略号 \ldots, \cdots, \dots
                dots_pattern = r'\\(ldots|cdots|dots)(?![a-zA-Z])'
                repaired = re.sub(dots_pattern, r'$\\\1$', repaired)
                
                # 匹配指数表达式（如 x^2, a^{n+1}）
                exponent_pattern = r'\b([a-zA-Z]+)\^(\{[^}]+\}|\w+)\b'
                def fix_exponent(match):
                    base = match.group(1)
                    exponent = match.group(2)
                    if exponent.startswith('{') and exponent.endswith('}'):
                        return f'${base}^{exponent}$'
                    else:
                        return f'${base}^{{{exponent}}}$'
                repaired = re.sub(exponent_pattern, fix_exponent, repaired)
                
                # 匹配分数表达式（如 \frac{a}{b}）
                fraction_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
                repaired = re.sub(fraction_pattern, r'$\\frac{\1}{\2}$', repaired)
                
                # 匹配积分表达式（如 \int_0^1 x^2 dx）
                integral_pattern = r'\\int(_\{[^}]+\})?(\^\{[^}]+\})?\s*([^$\n]+)\s*d([a-zA-Z])'
                repaired = re.sub(integral_pattern, r'$$\\int\1\2 \3 d\4$$', repaired)
                
                # 清理可能的双重包装
                repaired = re.sub(r'\$\$([^$]+)\$\$', r'$$\1$$', repaired)  # 保持块级
                repaired = re.sub(r'\$\$([^$\n]+)\$', r'$\1$', repaired)    # 转换为行内
                
            self.logger.debug("Enhanced math format repair completed with auto-wrapping")
            return repaired
            
        except Exception as e:
            self.logger.error(f"Error in enhanced math repair: {e}")
            return text
    
    def _repair_code_format(self, text: str) -> str:
        """修复代码格式 - 增强版"""
        if not self.config.enable_code_repair:
            return text

        try:
            repaired = text

            # 1. 确保 ``` 独占一行且前后有换行，清理语言标识符
            # 将 ```python code... ``` 转换为 ```python\ncode...\n```
            def replacer(m):
                lang = m.group(1).strip().lower() if m.group(1) else 'text'
                code = m.group(2).strip()
                return f"```{lang}\n{code}\n```"
            
            # 这个正则处理 ```lang content ``` 在同一行的情况
            repaired = re.sub(r'```\s*(\S*)\s*([^\n`]+?)\s*```', replacer, repaired)

            # 2. 修复缺失语言标识的代码块，并确保换行
            # 匹配没有语言标识的 ```\n content \n```
            repaired = re.sub(r'(?<!`)```\n([\s\S]+?)\n```(?!`)', r'```text\n\1\n```', repaired)
            # 匹配紧凑的 ```content```
            repaired = re.sub(r'(?<!`)```([^\n`]+?)```(?!`)', r'```text\n\1\n```', repaired)


            # 3. 检查并修复未闭合的代码块
            if repaired.count('```') % 2 != 0:
                # 如果最后一个 ``` 后面有内容，则认为它是未闭合的
                last_marker = repaired.rfind('```')
                if last_marker != -1:
                    content_after_last_marker = repaired[last_marker + 3:]
                    if content_after_last_marker.strip():
                         if not repaired.endswith('\n'):
                            repaired += '\n'
                         repaired += '```'
                         self.logger.debug("Added missing code block closing tag at the end.")

            # 4. 修复行内代码 - 更保守的方式
            repaired = re.sub(r'`([^`\n]{1,100})$', r'`\1`', repaired, flags=re.MULTILINE)

            self.logger.debug("Enhanced code format repair completed")
            return repaired

        except Exception as e:
            self.logger.error(f"Error in enhanced code repair: {e}")
            return text
    
    def _repair_table_format(self, text: str) -> str:
        """Fixes markdown tables with inconsistent column counts."""
        # This regex finds potential markdown tables (header + separator + at least one row)
        table_regex = re.compile(
            r"(^\|(?:.*\|)+.*$\n"  # Header line
            r"^\s*\|(?:\s*:?-+:?\s*\|)+.*$\n"  # Separator line
            r"(?:^\|(?:.*\|)+.*$\n?)+)",  # One or more data rows
            re.MULTILINE
        )

        def fix_table_match(match):
            table_text = match.group(0)
            lines = table_text.strip().split('\n')
            
            if len(lines) < 2:
                return table_text # Not a valid table

            header = lines[0]
            # Split and filter out empty strings from start/end pipes
            header_cells = [c.strip() for c in header.strip().strip('|').split('|')]
            col_count = len(header_cells)
            
            if col_count == 0:
                return table_text # No columns found

            # Rebuild table with consistent column count
            new_table_lines = []
            
            # Add header
            new_table_lines.append('| ' + ' | '.join(header_cells) + ' |')
            
            # Add new separator
            new_table_lines.append('|' + ' --- |' * col_count)
            
            # Add data rows (from line 2 onwards, as line 1 is the original separator)
            for row_line in lines[2:]:
                cells = [c.strip() for c in row_line.strip().strip('|').split('|')]
                while len(cells) < col_count:
                    cells.append('') # Pad
                cells = cells[:col_count] # Truncate
                new_table_lines.append('| ' + ' | '.join(cells) + ' |')
            
            return '\n'.join(new_table_lines)

        try:
            return table_regex.sub(fix_table_match, text)
        except Exception as e:
            self.logger.warning(f"Failed to repair table format: {e}")
            return text

    def _repair_general_format(self, text: str) -> str:
        """通用格式修复 - 增强版Markdown格式修复"""
        repaired = text
        
        # 修复Markdown标题 - 增强版
        if self.config.enable_markdown_repair and self.config.markdown_fix_headers:
            # 修复标题格式：确保#后面有空格
            repaired = re.sub(r'^(#{1,6})([^#\s])', r'\1 \2', repaired, flags=re.MULTILINE)
            
            # 修复标题和表格在同一行的问题 (例如: ### title | col1 | col2 |)
            # 修复为: ### title\n| col1 | col2 |
            repaired = re.sub(r'^(#{1,6}\s+.*?)\s*\|', r'\1\n|', repaired, flags=re.MULTILINE)

        # 修复列表格式 - 增强版
        if self.config.enable_markdown_repair and self.config.markdown_fix_lists:
            # 无序列表
            repaired = re.sub(r'^(\s*)([*\-+])([^\s])', r'\1\2 \3', repaired, flags=re.MULTILINE)
            # 有序列表
            repaired = re.sub(r'^(\s*)(\d+\.)([^\s])', r'\1\2 \3', repaired, flags=re.MULTILINE)
            # 修复列表项之间的间距
            repaired = re.sub(r'(^\s*[*\-+\d+\.] .+)\n\n+(^\s*[*\-+\d+\.] )', r'\1\n\2', repaired, flags=re.MULTILINE)
        
        # 修复引用格式 - 增强版
        if self.config.enable_markdown_repair and self.config.markdown_fix_quotes:
            repaired = re.sub(r'^(>+)([^>\s])', r'\1 \2', repaired, flags=re.MULTILINE)
            # 修复多行引用
            repaired = re.sub(r'(^> .+)\n([^>\n])', r'\1\n> \2', repaired, flags=re.MULTILINE)
        
        # 修复链接格式 - 增强版
        if self.config.enable_markdown_repair and self.config.markdown_fix_links:
            # 标准链接格式
            repaired = re.sub(r'\[([^\]]+)\]\s*\(([^)]+)\)', r'[\1](\2)', repaired)
            # 修复不完整的链接
            repaired = re.sub(r'\[\]\(([^)]+)\)', r'[\1](\1)', repaired)
        
        # 修复粗体和斜体格式
        if self.config.enable_markdown_repair:
            # 修复不完整的粗体格式
            repaired = re.sub(r'\*\*([^*]+?)\*(?!\*)', r'**\1**', repaired)
            repaired = re.sub(r'(?<!\*)\*([^*]+?)\*\*', r'**\1**', repaired)
            # 修复不完整的斜体格式（更保守）
            repaired = re.sub(r'(?<!\*)\*([^*\s]+(?:\s+[^*\s]+)*)\*(?!\*)', r'*\1*', repaired)
        
        # 修复代码块格式
        if self.config.enable_code_repair:
            # 修复不完整的代码块
            if repaired.count('```') % 2 != 0:
                repaired += '\n```'
            # 修复行内代码格式
            repaired = re.sub(r'`([^`\n]+)`', r'`\1`', repaired)
        
        # 修复表格格式
        if self.config.enable_markdown_repair:
            # 确保表格行的格式正确
            repaired = re.sub(r'^\|(.+)\|$', r'| \1 |', repaired, flags=re.MULTILINE)
            # 智能修复表格列数不一致的问题
            repaired = self._repair_table_format(repaired)

        # 修复段落换行：将非特殊行后的单个换行符视作段落分隔
        if self.config.enable_markdown_repair:
            # 将连续的多个换行符合并为两个，形成段落
            repaired = re.sub(r'\n{3,}', '\n\n', repaired)
            # 将前面不是特殊字符（如列表符、标题符）的单个换行符替换为两个，强制分段
            repaired = re.sub(r'(?<=[^\n#*\->|])\n(?=[^\n#*\->|])', '\n\n', repaired)

        # 强制确保所有块级元素（标题、列表、引用、代码块、水平线）前后都有空行
        # 这是最健壮的修复方式，可以解决大部分Markdown渲染问题
        if self.config.enable_markdown_repair:
            # 识别所有主要的块级元素
            block_pattern = re.compile(
                r"^\s*(?:#{1,6}\s|[-*+]\s|\d+\.\s|>\s*|```|---|\|)"
            )

            lines = repaired.split('\n')
            result_lines = []
            if lines:
                for i, line in enumerate(lines):
                    is_block = block_pattern.match(line)
                    if is_block and i > 0 and result_lines and result_lines[-1].strip() != "":
                        # 如果当前行是块级元素，不是第一行，并且
                        # 前面的结果行不为空，则在此行前添加一个空行。
                        result_lines.append("")
                    
                    result_lines.append(line)
            
            repaired = "\n".join(result_lines)
            
            # 再次清理可能产生的多余空行
            repaired = re.sub(r'\n{3,}', '\n\n', repaired)

        return repaired

    def _repair_resume_format(self, text: str) -> str:
        """
        修复简历格式
        """
        if not self.config.enable_resume_repair:
            return text

        # 添加标题
        if not text.strip().startswith("# 个人简历"):
            text = f"# 个人简历\\n\\n{text}"

        # 加粗主要部分
        text = re.sub(r"^(联系方式|个人总结|工作经验|教育背景|专业技能|软技能|荣誉与奖项)", r"**\\1**", text, flags=re.MULTILINE)

        # 格式化列表
        text = re.sub(r"^\s*([·*•-])\s*(.*)", r"- \\2", text, flags=re.MULTILINE)

        # 添加结尾
        if "---" not in text:
            text += "\\n\\n---\\n*这份简历由AI优化，祝您求职顺利！*"

        return text

    def _final_cleanup(self, text: str) -> str:
        """最终清理 - 已被安全版本替代"""
        return self._safe_final_cleanup(text, text)
    
    def create_structured_output(self, content: str, output_type: str = "general") -> Dict[str, Any]:
        """
        创建结构化输出格式
        
        Args:
            content: 修复后的内容
            output_type: 输出类型
            
        Returns:
            结构化的输出字典
        """
        return {
            "type": "ai_response",
            "content": content,
            "output_type": output_type,
            "format_version": "1.0",
            "timestamp": self._get_current_time(),
            "metadata": {
                "repaired": True,
                "original_length": len(content),
                "repair_rules_applied": [
                    rule for rule in ['json_repair', 'math_repair', 'markdown_repair', 'code_repair', 'structure_repair']
                    if getattr(self.config, f'enable_{rule}', False)
                ],
                "correction_intensity": self.config.correction_intensity.value,
                "config_version": "1.0"
            }
        }
    
    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.utcnow().isoformat() + "Z"
    
    def detect_output_type(self, text: str) -> str:
        """
        自动检测输出类型 - 改进的检测逻辑
        
        Args:
            text: 输入文本
            
        Returns:
            检测到的输出类型
        """
        text_lower = text.lower().strip()

        # 1. 优先检测严格的格式：代码块
        if text_lower.startswith('```') and text_lower.endswith('```'):
            # 检查是否是json代码块
            if text_lower.startswith('```json'):
                return "json"
            return "code"

        # 2. 其次检测严格的数学公式环境
        if (text_lower.startswith('$$') and text_lower.endswith('$$')) or \
           (text_lower.startswith('\\[') and text_lower.endswith('\\]')) or \
           (text_lower.startswith('\\(') and text_lower.endswith('\\)')):
            return "math"

        # 3. 检测JSON：更严格的规则
        # 必须以 { 或 [ 开头，并以 } 或 ] 结尾
        if text_lower.startswith('{') or text_lower.startswith('['):
            # 排除包含明显 LaTeX 命令的伪 JSON
            # 如果看起来像JSON，但包含LaTeX关键字，则更有可能是数学公式
            latex_keywords = ['\\frac', '\\sum', '\\int', '\\lim', '\\sqrt', '\\{', '\\}']
            if any(keyword in text_lower for keyword in latex_keywords):
                return "math"
            return "json"

        # 4. 基于内容的模糊检测
        # 检测代码
        code_indicators = ['def ', 'function', 'class ', 'public static', 'import ']
        if '```' in text_lower or any(indicator in text_lower for indicator in code_indicators):
            return "code"
        
        # 检测数学内容 - 更保守的策略
        # 只有当包含明显的数学标记时才判断为数学类型
        math_indicators = ['\\frac', '\\sqrt', '\\sum', '\\int', '\\lim', '\\alpha', '\\beta', '\\gamma', '\\pi', '\\theta']
        if any(indicator in text_lower for indicator in math_indicators):
            return "math"
        
        # 检测计算过程（包含等式和步骤）
        if ('=' in text and ('步' in text or 'step' in text_lower)) or \
           ('计算' in text and ('=' in text or '+' in text or '-' in text or '*' in text or '/' in text)):
            return "math"
        
        # 5. 默认返回通用类型
        return "general"
    
    def batch_repair(self, texts: List[str]) -> List[str]:
        """
        批量修复多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            修复后的文本列表
        """
        repaired_texts = []
        for text in texts:
            output_type = self.detect_output_type(text)
            repaired = self.repair_ai_output(text, output_type)
            repaired_texts.append(repaired)
        
        return repaired_texts

# 全局实例
format_repair_service = AIOutputFormatRepair()