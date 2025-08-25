#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试格式修复集成功能
验证stream_processor中的格式修复是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eztalk_proxy.services.format_repair import format_repair_service
from eztalk_proxy.services.stream_processor import preprocess_ai_output_content, postprocess_ai_output_chunk

def test_code_block_repair_detailed():
    """详细测试代码块格式修复"""
    print("=== 详细测试代码块格式修复 ===")
    
    # 测试用例1: 不完整的代码块
    test_content = "```bash\n# Debian/Ubuntu 举例 curl -fsSL https://pkg.cloudclient.com/install.deb.sh"
    
    print("原始内容:")
    print(repr(test_content))
    print("\n原始内容显示:")
    print(test_content)
    
    # 直接测试格式修复服务
    print("\n--- 直接测试格式修复服务 ---")
    repaired_direct = format_repair_service.repair_ai_output(test_content, "code")
    print("直接修复结果:", repr(repaired_direct))
    
    if repaired_direct != test_content:
        print("✅ 直接修复成功")
        print("修复后显示:")
        print(repaired_direct)
    else:
        print("❌ 直接修复失败")
    
    # 测试_repair_code_format方法
    print("\n--- 测试_repair_code_format方法 ---")
    try:
        repaired_code = format_repair_service._repair_code_format(test_content)
        print("_repair_code_format结果:", repr(repaired_code))
        
        if repaired_code != test_content:
            print("✅ _repair_code_format成功")
        else:
            print("❌ _repair_code_format失败")
    except Exception as e:
        print(f"❌ _repair_code_format异常: {e}")
    
    # 测试配置
    print("\n--- 检查配置 ---")
    config = format_repair_service.config
    print(f"enable_code_repair: {config.enable_code_repair}")
    print(f"code_fix_incomplete_blocks: {config.code_fix_incomplete_blocks}")
    print(f"correction_intensity: {config.correction_intensity}")
    
    # 测试代码块计数
    print("\n--- 测试代码块计数 ---")
    import re
    code_block_count = len(re.findall(r'```', test_content))
    print(f"代码块标记数量: {code_block_count}")
    print(f"是否为奇数（需要修复）: {code_block_count % 2 != 0}")
    
    return repaired_direct

def test_safe_content_check():
    """测试安全内容检查"""
    print("\n=== 测试安全内容检查 ===")
    
    test_content = "```bash\n# Debian/Ubuntu 举例 curl -fsSL https://pkg.cloudclient.com/install.deb.sh"
    
    # 测试_is_safe_content_that_should_not_be_modified方法
    try:
        is_safe = format_repair_service._is_safe_content_that_should_not_be_modified(test_content)
        print(f"内容是否被认为是安全的（不需要修复）: {is_safe}")
        
        if is_safe:
            print("⚠️ 内容被认为是安全的，可能不会被修复")
        else:
            print("✅ 内容被认为需要修复")
    except Exception as e:
        print(f"❌ 安全检查异常: {e}")

def test_step_by_step_repair():
    """逐步测试修复过程"""
    print("\n=== 逐步测试修复过程 ===")
    
    test_content = "```bash\n# Debian/Ubuntu 举例 curl -fsSL https://pkg.cloudclient.com/install.deb.sh"
    
    print("1. 原始内容:", repr(test_content))
    
    # 步骤1: 安全检查
    is_safe = format_repair_service._is_safe_content_that_should_not_be_modified(test_content)
    print(f"2. 安全检查结果: {is_safe}")
    
    if is_safe:
        print("   ⚠️ 内容被认为安全，修复可能被跳过")
        return test_content
    
    # 步骤2: 基础清理
    try:
        cleaned = format_repair_service._safe_basic_format_cleanup(test_content)
        print(f"3. 基础清理结果: {repr(cleaned)}")
    except Exception as e:
        print(f"3. 基础清理异常: {e}")
        cleaned = test_content
    
    # 步骤3: 代码修复
    try:
        code_repaired = format_repair_service._safe_repair_code_format(cleaned, test_content)
        print(f"4. 代码修复结果: {repr(code_repaired)}")
        
        if code_repaired != cleaned:
            print("   ✅ 代码修复生效")
        else:
            print("   ❌ 代码修复未生效")
    except Exception as e:
        print(f"4. 代码修复异常: {e}")
        code_repaired = cleaned
    
    return code_repaired

if __name__ == "__main__":
    print("开始详细测试格式修复集成功能...\n")
    
    try:
        # 详细测试
        test_code_block_repair_detailed()
        
        # 安全内容检查
        test_safe_content_check()
        
        # 逐步测试
        test_step_by_step_repair()
        
        print("\n✅ 所有测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()