"""
测试 AGrail4Agent 评估设置
用于验证数据加载和基本功能
"""

import os
import sys

def test_data_files():
    """测试数据文件是否存在"""
    print("检查数据文件...")
    
    data_files = [
        "../../../data/agentharm/harmful.json",
        "../../../data/agentharm/benign.json",
        "../../../data/ASB/harmful.json",
        "../../../data/ASB/benign.json",
        "../../../data/R-Judge/harmful.json",
        "../../../data/R-Judge/benign.json",
    ]
    
    all_exist = True
    for file_path in data_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def test_imports():
    """测试必要的模块导入"""
    print("\n检查模块导入...")
    
    modules = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("torch", None),
        ("tqdm", None),
        ("utils", None),
        ("guardrail", None),
        ("prompts", None),
    ]
    
    all_imported = True
    for module_name, alias in modules:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_imported = False
    
    return all_imported


def test_data_loading():
    """测试数据加载功能"""
    print("\n测试数据加载...")
    
    try:
        from evaluate_defend import load_agentharm_data, load_asb_data, load_rjudge_data
        
        # 测试 AgentHarm
        try:
            data = load_agentharm_data("harmful")
            print(f"  ✓ AgentHarm harmful 数据加载成功 ({len(data)} 条)")
        except Exception as e:
            print(f"  ✗ AgentHarm harmful 数据加载失败: {e}")
            return False
        
        # 测试 ASB
        try:
            data = load_asb_data("benign")
            print(f"  ✓ ASB benign 数据加载成功 ({len(data)} 条)")
        except Exception as e:
            print(f"  ✗ ASB benign 数据加载失败: {e}")
            return False
        
        # 测试 R-Judge
        try:
            data = load_rjudge_data("harmful")
            print(f"  ✓ R-Judge harmful 数据加载成功 ({len(data)} 条)")
        except Exception as e:
            print(f"  ✗ R-Judge harmful 数据加载失败: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"  ✗ 无法导入评估模块: {e}")
        return False


def test_directories():
    """测试必要的目录"""
    print("\n检查/创建目录...")
    
    directories = [
        "result/defend_comparison",
        "logs/defend_comparison",
        "memory",
    ]
    
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"  + 创建目录: {dir_path}")
        else:
            print(f"  ✓ 目录已存在: {dir_path}")
    
    return True


def main():
    print("="*60)
    print("AGrail4Agent DEFEND 评估设置测试")
    print("="*60)
    
    results = {}
    
    # 测试数据文件
    results["data_files"] = test_data_files()
    
    # 测试模块导入
    results["imports"] = test_imports()
    
    # 测试数据加载
    results["data_loading"] = test_data_loading()
    
    # 测试目录
    results["directories"] = test_directories()
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ 所有测试通过！可以开始评估。")
        print("\n运行命令:")
        print("  bash run_defend_evaluation.sh")
        print("\n或单独测试:")
        print("  python evaluate_defend.py --dataset agentharm --data_type harmful --debug")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查上述错误。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
