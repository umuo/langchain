#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/6/25 16:21
from typing import TypedDict, List, Optional
from langgraph.graph import Graph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import subprocess
import tempfile
import os
import json


class CodeState(TypedDict):
    """状态定义"""
    user_query: str
    conversation_history: List[dict]
    generated_code: Optional[str]
    execution_result: Optional[str]
    error_message: Optional[str]
    iteration_count: int
    max_iterations: int
    is_complete: bool


class ClaudeCodeAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
            base_url="https://newapi.lacknb.com/v1",
            verbose=True,
            temperature=0
        )

    def analyze_query(self, state: CodeState) -> CodeState:
        """分析用户查询，理解需求"""
        system_prompt = """你是一个专业的编程助手。分析用户的需求，确定需要什么样的代码。

        请分析以下查询：
        1. 用户想要实现什么功能？
        2. 需要使用什么编程语言？
        3. 是否需要安装特定的依赖？
        4. 代码的复杂程度如何？

        请提供简洁的分析结果。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_query"])
        ]

        response = self.llm.invoke(messages)

        # 更新对话历史
        state["conversation_history"].append({
            "role": "analysis",
            "content": response.content
        })

        return state

    def generate_code(self, state: CodeState) -> CodeState:
        """生成代码"""
        system_prompt = """你是一个专业的程序员。基于用户的需求生成完整、可执行的代码。

        要求：
        1. 代码必须完整且可以直接运行
        2. 包含必要的错误处理
        3. 添加适当的注释
        4. 如果需要外部库，在代码开头注释说明
        5. 确保代码遵循最佳实践

        请只返回代码，不要包含其他解释文字。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"需求：{state['user_query']}")
        ]

        # 如果有之前的错误，包含错误信息
        if state["error_message"]:
            messages.append(
                HumanMessage(content=f"之前的代码执行出错：{state['error_message']}，请修复这些问题。")
            )

        response = self.llm.invoke(messages)

        # 提取代码（假设代码在```块中）
        code = self._extract_code(response.content)

        state["generated_code"] = code
        state["conversation_history"].append({
            "role": "generator",
            "content": response.content
        })

        return state

    def execute_code(self, state: CodeState) -> CodeState:
        """执行代码"""
        if not state["generated_code"]:
            state["error_message"] = "没有生成的代码可以执行"
            return state

        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(state["generated_code"])
                temp_file = f.name

            # 执行代码
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=30  # 30秒超时
            )

            # 清理临时文件
            os.unlink(temp_file)

            if result.returncode == 0:
                state["execution_result"] = result.stdout
                state["error_message"] = None
                state["is_complete"] = True
            else:
                state["error_message"] = result.stderr
                state["execution_result"] = None

        except subprocess.TimeoutExpired:
            state["error_message"] = "代码执行超时"
        except Exception as e:
            state["error_message"] = f"执行错误：{str(e)}"

        state["iteration_count"] += 1

        return state

    def review_and_improve(self, state: CodeState) -> CodeState:
        """审查代码并改进"""
        if state["is_complete"]:
            return state

        if state["iteration_count"] >= state["max_iterations"]:
            state["error_message"] = "已达到最大迭代次数，无法完成任务"
            return state

        system_prompt = """分析代码执行的错误，提供改进建议。

        请分析错误信息，确定问题所在：
        1. 是语法错误还是逻辑错误？
        2. 是否缺少必要的导入？
        3. 是否有环境依赖问题？
        4. 如何修复这些问题？

        请提供具体的修复建议。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            原始需求：{state['user_query']}
            生成的代码：
            ```python
            {state['generated_code']}
            ```
            错误信息：{state['error_message']}
            """)
        ]

        response = self.llm.invoke(messages)

        state["conversation_history"].append({
            "role": "reviewer",
            "content": response.content
        })

        return state

    def should_continue(self, state: CodeState) -> str:
        """决定下一步操作"""
        if state["is_complete"]:
            return "end"
        elif state["iteration_count"] >= state["max_iterations"]:
            return "end"
        elif state["error_message"]:
            return "improve"
        else:
            return "execute"

    def _extract_code(self, content: str) -> str:
        """从响应中提取代码"""
        lines = content.split('\n')
        code_lines = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip().startswith('```') and in_code_block:
                break
            elif in_code_block:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)
        else:
            # 如果没有找到代码块，返回整个内容
            return content


def create_claude_code_graph() -> Graph:
    """创建LangGraph工作流"""
    agent = ClaudeCodeAgent(api_key="sk-D6vgSJ0a8vgUZgUtaVbfnXkpxUiRVdcklZp14S62ZigNmGvi")

    # 创建图
    workflow = Graph()

    # 添加节点
    workflow.add_node("analyze", agent.analyze_query)
    workflow.add_node("generate", agent.generate_code)
    workflow.add_node("execute", agent.execute_code)
    workflow.add_node("improve", agent.review_and_improve)

    # 添加边
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("improve", "generate")

    # 添加条件边
    workflow.add_conditional_edges(
        "execute",
        agent.should_continue,
        {
            "end": END,
            "improve": "improve"
        }
    )

    # 设置入口点
    workflow.set_entry_point("analyze")

    return workflow.compile()


# 使用示例
def run_claude_code(user_query: str, api_key: str):
    """运行Claude Code功能"""

    # 创建工作流
    app = create_claude_code_graph()

    # 初始状态
    initial_state = CodeState(
        user_query=user_query,
        conversation_history=[],
        generated_code=None,
        execution_result=None,
        error_message=None,
        iteration_count=0,
        max_iterations=3,
        is_complete=False
    )

    # 执行工作流
    result = app.invoke(initial_state)

    return result


# 命令行接口
if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Claude Code - AI代码生成和执行工具")
    # parser.add_argument("query", help="描述你想要实现的功能")
    # parser.add_argument("--api-key", required=True, help="Anthropic API密钥")
    # parser.add_argument("--verbose", "-v", action="store_true", help="显示详细过程")
    #
    # args = parser.parse_args()

    # print(f"🚀 开始处理请求：{args.query}")
    print("-" * 50)

    result = run_claude_code("使用Python创建一个计算斐波那契数列的函数", "sk-D6vgSJ0a8vgUZgUtaVbfnXkpxUiRVdcklZp14S62ZigNmGvi")

    if result["is_complete"]:
        print("✅ 任务完成！")
        print(f"\n生成的代码：\n```python\n{result['generated_code']}\n```")
        if result["execution_result"]:
            print(f"\n执行结果：\n{result['execution_result']}")
    else:
        print("❌ 任务未能完成")
        if result["error_message"]:
            print(f"错误信息：{result['error_message']}")

    if args.verbose:
        print("\n对话历史：")
        for i, msg in enumerate(result["conversation_history"]):
            print(f"{i + 1}. [{msg['role']}] {msg['content'][:200]}...")


# 高级功能扩展
class AdvancedClaudeCodeAgent(ClaudeCodeAgent):
    """高级版本，支持更多功能"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.supported_languages = ['python', 'javascript', 'bash', 'sql']

    def detect_language(self, state: CodeState) -> CodeState:
        """检测需要使用的编程语言"""
        system_prompt = """分析用户查询，确定最适合的编程语言。

        支持的语言：Python, JavaScript, Bash, SQL

        请只返回语言名称，如：python"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_query"])
        ]

        response = self.llm.invoke(messages)
        language = response.content.strip().lower()

        state["language"] = language if language in self.supported_languages else "python"
        return state

    def install_dependencies(self, state: CodeState) -> CodeState:
        """安装必要的依赖"""
        # 分析代码中的import语句
        if not state["generated_code"]:
            return state

        import re
        imports = re.findall(r'import\s+(\w+)', state["generated_code"])

        # 常见的需要安装的包
        common_packages = {
            'requests': 'requests',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'sklearn': 'scikit-learn'
        }

        for imp in imports:
            if imp in common_packages:
                try:
                    subprocess.run(['pip', 'install', common_packages[imp]],
                                   check=True, capture_output=True)
                    print(f"已安装依赖：{common_packages[imp]}")
                except subprocess.CalledProcessError:
                    print(f"无法安装依赖：{common_packages[imp]}")

        return state


# 配置文件示例
CONFIG_EXAMPLE = """
# claude_code_config.yaml
anthropic:
  api_key: "your-api-key-here"
  model: "claude-3-sonnet-20240229"

execution:
  timeout: 30
  max_iterations: 3

supported_languages:
  - python
  - javascript
  - bash
  - sql

security:
  allow_file_operations: false
  allow_network_access: true
  restricted_imports:
    - os.system
    - subprocess.call
"""