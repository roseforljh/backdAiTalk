import sys
import os

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eztalk_proxy.services.format_repair import AIOutputFormatRepair

def run_tests():
    """运行格式修复服务的单元测试"""
    service = AIOutputFormatRepair()
    
    # --- 测试用例 ---
    test_cases = {
        "Simple Text": ("这是一段普通文本。", "general"),
        "Malformed JSON": ('{"name": "John", "age": 30,}', "json"),
        "LaTeX Math Formula": (r"e^{ix} = \cos x + i\sin x", "math"),
        "LaTeX in JSON-like structure": (r'{"formula": "\\sum_{n=0}^{\\infty} \\frac{x^n}{n!}"}', "math"),
        "Markdown with Code": ("这是一个代码块: ```python\nprint('hello')\n```", "code"),
        "Incomplete JSON": ('{"key": "value"', "json"),
        "Simple Markdown List": ("- 列表项1\n- 列表项2", "general"),
        "Complex LaTeX": (r"\[ \sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6} \]", "math"),
        "JSON code block": ('```json\n{"a": 1, "b": 2,}\n```', "json")
    }
    
    results = {}
    
    print("--- Running AI Output Format Repair Tests ---\n")
    
    for name, (text, expected_type) in test_cases.items():
        print(f"--- Testing: {name} ---")
        detected_type = service.detect_output_type(text)
        repaired_text = service.repair_ai_output(text, detected_type)
        
        type_check_passed = detected_type == expected_type
        
        results[name] = {
            "Original Text": text,
            "Expected Type": expected_type,
            "Detected Type": detected_type,
            "Repaired Text": repaired_text,
            "Type Check Passed": "✅" if type_check_passed else "❌"
        }
        
        print(f"Original Text: {text}")
        print(f"Expected Type: {expected_type}, Detected Type: {detected_type}")
        print(f"Type Check Passed: {'✅' if type_check_passed else '❌'}")
        print(f"Repaired Text: {repaired_text}\n")
        
    print("\n--- Test Summary ---")
    all_passed = True
    for name, result in results.items():
        status = result['Type Check Passed']
        print(f"{name}: {status}")
        if status == "❌":
            all_passed = False
            
    print("\n--- Overall Result ---")
    if all_passed:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed.")

if __name__ == "__main__":
    run_tests()