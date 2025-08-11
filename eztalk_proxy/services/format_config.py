"""
格式修复配置管理
提供灵活的配置选项来控制格式修复行为
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger("EzTalkProxy.Services.FormatConfig")

class CorrectionIntensity(Enum):
    """修复强度枚举"""
    LIGHT = "light"      # 轻度修复：只修复明显错误
    MEDIUM = "medium"    # 中度修复：标准修复规则
    STRONG = "strong"    # 强度修复：积极修复所有可能的问题

@dataclass
class FormatRepairConfig:
    """格式修复配置类"""
    
    # 基础开关 - 启用格式修复但避免重复处理
    enable_format_repair: bool = True
    enable_realtime_repair: bool = False  # 禁用实时修复，只在最终进行一次修复
    enable_final_repair: bool = True
    
    # 具体修复功能 - 使用保守设置
    enable_json_repair: bool = True
    enable_math_repair: bool = True
    enable_markdown_repair: bool = False  # 禁用Markdown修复以减少干扰
    enable_code_repair: bool = True
    enable_xml_html_repair: bool = False  # 禁用XML/HTML修复以减少干扰
    enable_structure_repair: bool = False  # 禁用结构修复以减少干扰
    enable_resume_repair: bool = True,  # 启用简历修复
    
    # 修复强度 - 使用轻度修复避免过度处理
    correction_intensity: CorrectionIntensity = CorrectionIntensity.LIGHT,
    
    # 性能优化
    enable_caching: bool = True
    enable_performance_optimization: bool = True
    enable_async_processing: bool = False
    enable_progressive_correction: bool = True
    
    # 缓存配置
    max_cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # 性能限制
    max_processing_time_ms: int = 5000
    chunk_size_threshold: int = 10000
    max_text_length: int = 100000
    
    # 数学公式修复配置
    math_auto_wrap: bool = True
    math_strict_mode: bool = False
    math_preserve_spacing: bool = True
    inline_math_detection: bool = True
    preserve_inline_math_flow: bool = True
    avoid_math_forced_newlines: bool = True
    smart_math_context_awareness: bool = True
    
    # 代码修复配置
    code_auto_language_detection: bool = True
    code_preserve_indentation: bool = True
    code_fix_incomplete_blocks: bool = True
    
    # JSON修复配置
    json_strict_quotes: bool = True
    json_remove_trailing_commas: bool = True
    json_fix_brackets: bool = True
    
    # Markdown修复配置
    markdown_fix_headers: bool = False  # 禁用标题修复
    markdown_fix_lists: bool = False    # 禁用列表修复
    markdown_fix_links: bool = False    # 禁用链接修复
    markdown_fix_quotes: bool = False   # 禁用引用修复
    
    # 调试选项
    debug_mode: bool = False
    log_repair_actions: bool = True
    preserve_original_on_error: bool = True

class FormatConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'config', 
            'format_repair_config.json'
        )
        self.config = FormatRepairConfig()
        self.load_config()
    
    def load_config(self) -> None:
        """从文件加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 处理枚举类型
                if 'correction_intensity' in config_data:
                    try:
                        config_data['correction_intensity'] = CorrectionIntensity(
                            config_data['correction_intensity']
                        )
                    except ValueError:
                        logger.warning(f"Invalid correction_intensity value, using default")
                        config_data['correction_intensity'] = CorrectionIntensity.MEDIUM
                
                # 更新配置
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                    else:
                        logger.warning(f"Unknown config key: {key}")
                
                logger.info(f"Loaded format repair config from {self.config_file}")
            else:
                logger.info("Config file not found, using default configuration")
                self.save_config()  # 创建默认配置文件
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
    
    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # 转换为字典，处理枚举
            config_dict = asdict(self.config)
            config_dict['correction_intensity'] = self.config.correction_intensity.value
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved format repair config to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update_config(self, **kwargs) -> None:
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                # 处理枚举类型
                if key == 'correction_intensity' and isinstance(value, str):
                    try:
                        value = CorrectionIntensity(value)
                    except ValueError:
                        logger.warning(f"Invalid correction_intensity value: {value}")
                        continue
                
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def get_config(self) -> FormatRepairConfig:
        """获取当前配置"""
        return self.config
    
    def reset_to_defaults(self) -> None:
        """重置为默认配置"""
        self.config = FormatRepairConfig()
        logger.info("Reset configuration to defaults")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        config_dict = asdict(self.config)
        config_dict['correction_intensity'] = self.config.correction_intensity.value
        return config_dict
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 检查数值范围
            if self.config.max_cache_size < 0:
                logger.error("max_cache_size must be >= 0")
                return False
            
            if self.config.max_processing_time_ms < 100:
                logger.error("max_processing_time_ms must be >= 100")
                return False
            
            if self.config.chunk_size_threshold < 1000:
                logger.error("chunk_size_threshold must be >= 1000")
                return False
            
            if self.config.max_text_length < 1000:
                logger.error("max_text_length must be >= 1000")
                return False
            
            # 检查逻辑依赖
            if self.config.enable_caching and self.config.max_cache_size == 0:
                logger.warning("Caching enabled but max_cache_size is 0")
            
            if self.config.enable_async_processing and self.config.chunk_size_threshold > self.config.max_text_length:
                logger.warning("chunk_size_threshold > max_text_length may cause issues")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_repair_rules_for_type(self, content_type: str) -> Dict[str, bool]:
        """根据内容类型获取适用的修复规则"""
        base_rules = {
            'enable_structure_repair': self.config.enable_structure_repair,
        }
        
        if content_type == "math":
            return {
                **base_rules,
                'enable_math_repair': self.config.enable_math_repair,
                'math_auto_wrap': self.config.math_auto_wrap,
                'math_strict_mode': self.config.math_strict_mode,
                'inline_math_detection': self.config.inline_math_detection,
                'preserve_inline_math_flow': self.config.preserve_inline_math_flow,
                'avoid_math_forced_newlines': self.config.avoid_math_forced_newlines,
                'smart_math_context_awareness': self.config.smart_math_context_awareness,
            }
        elif content_type == "code":
            return {
                **base_rules,
                'enable_code_repair': self.config.enable_code_repair,
                'code_auto_language_detection': self.config.code_auto_language_detection,
                'code_fix_incomplete_blocks': self.config.code_fix_incomplete_blocks,
            }
        elif content_type == "json":
            return {
                **base_rules,
                'enable_json_repair': self.config.enable_json_repair,
                'json_strict_quotes': self.config.json_strict_quotes,
                'json_remove_trailing_commas': self.config.json_remove_trailing_commas,
            }
        else:  # general
            return {
                **base_rules,
                'enable_markdown_repair': self.config.enable_markdown_repair,
                'enable_math_repair': self.config.enable_math_repair,
                'enable_code_repair': self.config.enable_code_repair,
                'markdown_fix_headers': self.config.markdown_fix_headers,
                'markdown_fix_lists': self.config.markdown_fix_lists,
            }

# 全局配置管理器实例
config_manager = FormatConfigManager()

def get_format_config() -> FormatRepairConfig:
    """获取全局格式修复配置"""
    return config_manager.get_config()

def update_format_config(**kwargs) -> None:
    """更新全局格式修复配置"""
    config_manager.update_config(**kwargs)

def reload_format_config() -> None:
    """重新加载配置"""
    config_manager.load_config()