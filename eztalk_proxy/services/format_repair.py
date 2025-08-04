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
        
        # 数学公式修复配置
        self.math_patterns = {
            # 常见数学表达式
            'exponential': re.compile(r'([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)'),
            'fraction': re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}'),
            'sqrt': re.compile(r'\\sqrt\{([^}]+)\}'),
            'subscript': re.compile(r'([a-zA-Z0-9]+)_([a-zA-Z0-9]+)'),
            'formula': re.compile(r'([a-zA-Z]+)\s*=\s*(.+?)(?=\n|$)'),
        }
    
    def repair_ai_output(self, text: str, output_type: str = "general") -> str:
        """
        修复AI输出格式
        
        Args:
            text: 原始AI输出文本
            output_type: 输出类型 ("general", "math", "code", "json")
            
        Returns:
            修复后的文本
        """
        if not text or not text.strip():
            return text
            
        # 检查是否启用格式修复
        if not self.config.enable_format_repair:
            return text
            
        # 检查文本长度限制
        if len(text) > self.config.max_text_length:
            self.logger.warning(f"Text length {len(text)} exceeds limit {self.config.max_text_length}")
            return text
            
        try:
            repaired_text = text
            
            # 1. 基础格式清理
            repaired_text = self._basic_format_cleanup(repaired_text)
            
            # 2. 根据输出类型进行专门修复
            if output_type == "json" and self.config.enable_json_repair:
                repaired_text = self._repair_json_format(repaired_text)
            elif output_type == "math" and self.config.enable_math_repair:
                repaired_text = self._repair_math_format(repaired_text)
            elif output_type == "code" and self.config.enable_code_repair:
                repaired_text = self._repair_code_format(repaired_text)
            else:
                # 通用修复
                repaired_text = self._repair_general_format(repaired_text)
            
            # 3. 最终清理
            repaired_text = self._final_cleanup(repaired_text)
            
            return repaired_text
            
        except Exception as e:
            self.logger.error(f"Error repairing AI output: {e}")
            return text  # 出错时返回原始文本
    
    def _basic_format_cleanup(self, text: str) -> str:
        """基础格式清理"""
        # 移除多余的空行（保留最多2个连续换行）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 移除行尾空白
        text = re.sub(r'[ \t]+\n', '\n', text)
        
        # 修复中英文混排的空格问题 - 更精确的匹配
        # 只在中文和英文字母/数字之间添加空格，避免过度处理
        text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z0-9])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fa5])', r'\1 \2', text)
        
        return text.strip()
    
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
    
    def _repair_math_format(self, text: str) -> str:
        """修复数学公式格式 - 更保守的修复策略"""
        if not self.config.enable_math_repair:
            return text
            
        try:
            repaired = text
            
            # 检查是否已经包含数学符号，如果有则谨慎处理
            has_math_delimiters = any(delimiter in repaired for delimiter in ['$', '\\[', '\\]', '\\(', '\\)'])
            
            if not has_math_delimiters:
                # 只对明显的数学表达式进行修复，避免破坏普通文本
                
                # 修复明显的数学公式（如勾股定理）
                repaired = re.sub(r'\b([a-zA-Z])\^([0-9]+)\s*\+\s*([a-zA-Z])\^([0-9]+)\s*=\s*([a-zA-Z])\^([0-9]+)\b',
                                 r'$\1^{\2} + \3^{\4} = \5^{\6}$', repaired)
                
                # 修复单独的指数表达式（更严格的条件）
                repaired = re.sub(r'\b([a-zA-Z])\^([0-9]+)\b(?!\w)', r'$\1^{\2}$', repaired)
                
                # 修复欧拉公式等科学公式
                repaired = re.sub(r'\be\^([^\s\+\-\*\/\=\)\.]+)', r'$e^{\1}$', repaired)
            
            # 修复破损的LaTeX语法
            # 修复缺失的大括号
            repaired = re.sub(r'\\frac\s*([^{])', r'\\frac{\1', repaired)
            repaired = re.sub(r'\\sqrt\s*([^{])', r'\\sqrt{\1', repaired)
            
            # 修复破损的分数表达式
            repaired = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}(?!\})', lambda m: f'\\frac{{{m.group(1)}}}{{{m.group(2)}}}', repaired)
            
            # 清理多余的转义字符
            repaired = re.sub(r'\\+([{}])', r'\\\1', repaired)
            
            # 修复破损的数学分隔符
            repaired = re.sub(r'\\\}\s*-\s*\\\]', '', repaired)  # 修复您截图中的错误
            repaired = re.sub(r'\{(\d+)\}\{(\d+)\s*imes\s*(\d+)\}', r'{\1 \times \2 \times \3}', repaired)  # 修复乘法表达式
            
            self.logger.debug("Math format repair completed")
            return repaired
            
        except Exception as e:
            self.logger.error(f"Error in math repair: {e}")
            return text
    
    def _repair_code_format(self, text: str) -> str:
        """修复代码格式"""
        if not self.config.enable_code_repair:
            return text
            
        try:
            repaired = text
            
            # 修复不完整的代码块
            repaired = re.sub(r'```(\w*)\n(.*?)(?!```)', r'```\1\n\2\n```', repaired, flags=re.DOTALL)
            
            # 修复行内代码
            repaired = re.sub(r'`([^`\n]+)(?!`)', r'`\1`', repaired)
            
            # 修复缺失语言标识的代码块
            repaired = re.sub(r'```\n(.*?)\n```', r'```text\n\1\n```', repaired, flags=re.DOTALL)
            
            self.logger.debug("Code format repair completed")
            return repaired
            
        except Exception as e:
            self.logger.error(f"Error in code repair: {e}")
            return text
    
    def _repair_general_format(self, text: str) -> str:
        """通用格式修复"""
        repaired = text
        
        # 修复Markdown标题
        if self.config.enable_markdown_repair and self.config.markdown_fix_headers:
            repaired = re.sub(r'^(#{1,6})([^#\s])', r'\1 \2', repaired, flags=re.MULTILINE)
        
        # 修复列表格式
        if self.config.enable_markdown_repair and self.config.markdown_fix_lists:
            repaired = re.sub(r'^(\s*)([*\-+])([^\s])', r'\1\2 \3', repaired, flags=re.MULTILINE)
            repaired = re.sub(r'^(\s*)(\d+\.)([^\s])', r'\1\2 \3', repaired, flags=re.MULTILINE)
        
        # 修复引用格式
        if self.config.enable_markdown_repair and self.config.markdown_fix_quotes:
            repaired = re.sub(r'^(>+)([^>\s])', r'\1 \2', repaired, flags=re.MULTILINE)
        
        # 修复链接格式
        if self.config.enable_markdown_repair and self.config.markdown_fix_links:
            repaired = re.sub(r'\[([^\]]+)\]\s*\(([^)]+)\)', r'[\1](\2)', repaired)
        
        # 数学和代码混合修复
        if self.config.enable_math_repair:
            repaired = self._repair_math_format(repaired)
            
        if self.config.enable_code_repair:
            repaired = self._repair_code_format(repaired)
        
        return repaired
    
    def _final_cleanup(self, text: str) -> str:
        """最终清理"""
        # 移除多余的空白字符
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 确保段落间有适当间距
        text = re.sub(r'([.!?。！？])\s*\n([A-Z\u4e00-\u9fa5])', r'\1\n\n\2', text)
        
        # 最终去除首尾空白
        return text.strip()
    
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