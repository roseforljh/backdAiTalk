#!/usr/bin/env python3
"""
数学公式修复功能测试脚本
用于验证AI输出格式修复系统对数学内容的处理效果
"""

import sys
import os

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from eztalk_proxy.services.format_repair import AIOutputFormatRepair

def test_math_fixes():
    """测试数学公式修复功能"""
    service = AIOutputFormatRepair()
    
    print("🔧 AI输出格式修复系统 - 数学内容测试")
    print("=" * 60)
    
    # 测试用例 - 模拟您截图中遇到的问题
    test_cases = [
        {
            "name": "破损的LaTeX语法修复",
            "input": "计算 \\} - \\] {7}{57 imes 2}{5 2}",
            "expected_fix": "清理破损的LaTeX符号"
        },
        {
            "name": "计算过程格式修复", 
            "input": "计算\n\n第一步，计算乘法：\n\n第二步，进行减法：\n\\} - \\]",
            "expected_fix": "保持计算步骤结构"
        },
        {
            "name": "分数公式修复",
            "input": "1/4*6-7/5等于几",
            "expected_fix": "修复为 $\\frac{1}{4} \\times 6 - \\frac{7}{5}$"
        },
        {
            "name": "简单数学表达式",
            "input": "找到公分母，分母分别是 2 和 5，公分母为 10。",
            "expected_fix": "保持文本格式不变"
        },
        {
            "name": "勾股定理公式",
            "input": "a^2 + b^2 = c^2",
            "expected_fix": "$a^{2} + b^{2} = c^{2}$"
        },
        {
            "name": "混合数学内容",
            "input": "把两个分数都转成以 10 为分母：\n{7}{57 imes 2}{5 2} ={14}[] 相减：[",
            "expected_fix": "修复破损的大括号和LaTeX语法"
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 测试 {i}: {case['name']}")
        print("-" * 40)
        
        # 检测输出类型
        detected_type = service.detect_output_type(case['input'])
        print(f"🔍 检测类型: {detected_type}")
        
        # 应用修复
        repaired = service.repair_ai_output(case['input'], detected_type)
        
        print(f"📝 原始内容: {repr(case['input'])}")
        print(f"✨ 修复后: {repr(repaired)}")
        print(f"💡 预期效果: {case['expected_fix']}")
        
        # 检查修复效果
        improvement_detected = len(repaired.strip()) > 0 and repaired != case['input']
        
        results.append({
            'name': case['name'],
            'original': case['input'],
            'repaired': repaired,
            'type': detected_type,
            'improved': improvement_detected
        })
        
        status = "✅ 已修复" if improvement_detected else "⚠️  无变化"
        print(f"📊 状态: {status}")
    
    # 总结报告
    print("\n" + "=" * 60)
    print("📈 测试总结报告")
    print("=" * 60)
    
    total_tests = len(results)
    improved_tests = sum(1 for r in results if r['improved'])
    
    print(f"📊 总测试数: {total_tests}")
    print(f"✅ 成功修复: {improved_tests}")
    print(f"⚠️  无变化: {total_tests - improved_tests}")
    print(f"📈 修复率: {(improved_tests/total_tests)*100:.1f}%")
    
    # 详细结果
    print(f"\n📋 详细结果:")
    for result in results:
        status = "✅" if result['improved'] else "⚠️"
        print(f"  {status} {result['name']} (类型: {result['type']})")
    
    # 特殊测试：批量修复
    print(f"\n🔄 批量修复测试:")
    batch_texts = [case['input'] for case in test_cases[:3]]
    batch_results = service.batch_repair(batch_texts)
    
    print(f"📊 批量处理了 {len(batch_texts)} 个文本")
    print(f"✅ 批量修复完成")
    
    return results

def test_performance():
    """测试性能"""
    print(f"\n⚡ 性能测试")
    print("-" * 40)
    
    service = AIOutputFormatRepair()
    
    # 测试大文本处理
    large_text = "计算 1/4*6-7/5等于几\n" * 100
    
    import time
    start_time = time.time()
    result = service.repair_ai_output(large_text, "math")
    end_time = time.time()
    
    processing_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    print(f"📊 文本长度: {len(large_text)} 字符")
    print(f"⏱️  处理时间: {processing_time:.2f} 毫秒")
    
    if processing_time > 0:
        chars_per_second = len(large_text) / (processing_time / 1000)
        print(f"🚀 处理速度: {chars_per_second:,.0f} 字符/秒")
        
        if chars_per_second > 100000:  # 10万字符/秒
            print("✅ 性能优秀")
        elif chars_per_second > 50000:  # 5万字符/秒
            print("⚠️  性能良好")
        else:
            print("❌ 性能需要优化")
    else:
        print("🚀 处理速度: 极快 (< 1毫秒)")
        print("✅ 性能优秀")

if __name__ == "__main__":
    try:
        print("🚀 开始测试AI输出格式修复系统...")
        
        # 运行主要测试
        results = test_math_fixes()
        
        # 运行性能测试
        test_performance()
        
        print(f"\n🎉 测试完成!")
        print("💡 提示: 如果发现问题，请检查后端格式修复配置")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)