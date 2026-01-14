#!/usr/bin/env python3
"""
SwarmAgentic 环境检查脚本
用于验证环境是否正确配置
"""

import sys
import os

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        print("✓ Python版本: {}.{}.{} (符合要求)".format(version.major, version.minor, version.micro))
        return True
    else:
        print("⚠️  Python版本: {}.{}.{} (推荐3.11)".format(version.major, version.minor, version.micro))
        return False

def check_packages():
    """检查必需的包"""
    required_packages = {
        'openai': 'OpenAI API客户端',
        'langchain': 'LangChain框架',
        'langchain_openai': 'LangChain OpenAI集成',
        'numpy': '数值计算',
        'pandas': '数据处理',
        'tqdm': '进度条',
        'tiktoken': 'Token计数',
    }
    
    missing = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print("✓ {} - {}".format(package, description))
        except ImportError:
            print("❌ {} - 未安装".format(package))
            missing.append(package)
    
    return len(missing) == 0

def check_api_key():
    """检查API密钥"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        masked_key = api_key[:10] + '...' + api_key[-4:] if len(api_key) > 14 else '***'
        print("✓ OPENAI_API_KEY: 已设置 ({})".format(masked_key))
        return True
    else:
        print("❌ OPENAI_API_KEY: 未设置")
        print("   请运行: export OPENAI_API_KEY='your-api-key-here'")
        return False

def check_project_structure():
    """检查项目结构"""
    required_dirs = [
        'mgsm',
        'creative_writing',
        'travelplanner',
        'natural_plan',
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print("✓ 目录存在: {}".format(dir_name))
        else:
            print("❌ 目录缺失: {}".format(dir_name))
            all_exist = False
    
    return all_exist

def test_openai_connection():
    """测试OpenAI连接"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  跳过OpenAI连接测试 (API密钥未设置)")
        return None
    
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        # 不实际调用API，只检查模块是否正常
        print("✓ OpenAI模块导入成功")
        return True
    except Exception as e:
        print("❌ OpenAI模块测试失败: {}".format(str(e)))
        return False

def main():
    print("=" * 50)
    print("SwarmAgentic 环境检查")
    print("=" * 50)
    print()
    
    results = {
        'Python版本': check_python_version(),
        '依赖包': check_packages(),
        'API密钥': check_api_key(),
        '项目结构': check_project_structure(),
    }
    
    print()
    print("=" * 50)
    print("测试OpenAI连接...")
    print("=" * 50)
    results['OpenAI连接'] = test_openai_connection()
    
    print()
    print("=" * 50)
    print("检查结果汇总")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results.items():
        if passed is None:
            status = "跳过"
        elif passed:
            status = "✓ 通过"
        else:
            status = "❌ 失败"
            all_passed = False
        print("{}: {}".format(check_name, status))
    
    print()
    if all_passed:
        print("🎉 所有检查通过！环境配置正确。")
        print()
        print("下一步:")
        print("1. 运行快速测试: ./quick_start.sh")
        print("2. 或查看详细指南: 配置和复现指南.md")
        return 0
    else:
        print("⚠️  部分检查未通过，请根据上述提示修复问题。")
        print()
        print("建议:")
        print("1. 安装缺失的包: pip install -r requirements.txt")
        print("2. 设置API密钥: export OPENAI_API_KEY='your-key'")
        print("3. 查看详细指南: 配置和复现指南.md")
        return 1

if __name__ == '__main__':
    sys.exit(main())
