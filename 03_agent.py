# -*- coding: utf-8 -*-
"""
============================================================================
模块 3（LangChain 版）：用 LangChain 的 @tool + bind_tools 实现 ReAct Agent
============================================================================

【为什么改成 LangChain 版】
    前一版用纯 openai 库 + 正则从文本里抠 "Action:"，是“手写 ReAct 原教旨”，
    用来让你看清每一行。但实习/真实项目里不会这么写——LangChain 已经把
    “工具定义、模型调用、参数解析”都封装好了。

    本版本用 LangChain 的标准写法：
      - @tool       ：把普通函数变成“带 schema 的工具”（模型知道它长啥样、怎么填参）
      - bind_tools  ：把工具挂到模型上，模型走标准 tool_calling 协议返回结构化调用
                      （不再是文本里抠 Action，而是 response.tool_calls 字段）
      - ToolMessage ：工具结果用 LangChain 专用消息类型塞回对话（带 tool_call_id）

    ⚠️ 关于 LangGraph：LangChain 1.x 的官方 Agent 接口（create_agent）内部就是
      LangGraph。你之前说“暂时不用 LangGraph”，所以这里【循环仍手写】，但工具/模型
      全是 LangChain 框架。如果你想要“零手写循环的纯框架版”，那就得用 create_agent，
      那会引入 LangGraph——到时候说一声我换。

【它在项目中的位置】
    和前一版 03_agent.py 是【同一模块】的不同实现：大脑+手+回路都在，只是
    “手”和“调用方式”换成 LangChain 抽象。02_tools.py 的纯函数逻辑被复用。
============================================================================
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---- LangChain 组件 ----
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# ---- 复用模块2的纯函数逻辑（单一数据源，不重复造轮子） ----
import importlib.util
_module_path = Path(__file__).resolve().parent / "02_tools.py"
_spec = importlib.util.spec_from_file_location("tools02", str(_module_path))
tools02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tools02)
stat_calc = tools02.stat_calc
analyze_metric = tools02.analyze_metric
build_report = tools02.build_report


# ---------------------------------------------------------------------------
# 第 1 步：用 @tool 把纯函数包装成 LangChain 工具
# ---------------------------------------------------------------------------
# @tool 装饰器会读取：函数名、参数签名、docstring，
# 自动生成一份“工具 schema”给模型看（模型据此知道怎么调、填什么参数）。
# docstring 就是给模型看的“工具说明书”——比前一版手写 TOOL_GUIDE 更规范。
@tool
def stat_calc_tool(metric: str, operation: str = "差值") -> str:
    """计算人物A和人物B在某项指标上的统计结果。
    metric: 指标名，如 "微笑次数"、"说话时长秒"。
    operation: 计算方式，可选 "差值"/"比例"/"平均值"/"总和"，默认"差值"。
    """
    return stat_calc(metric, operation)


@tool
def analyze_metric_tool(metric: str) -> str:
    """分析某项指标：谁的数值更高、绝对差多少、差异程度（较小/中等/明显）。
    metric: 指标名，如 "微笑次数"、"打断次数"。
    """
    return analyze_metric(metric)


@tool
def build_report_tool() -> str:
    """汇总所有指标，生成一份行为数据互动分析草稿，供最终报告使用。无参数。"""
    return build_report()


# 工具清单 + 名称→工具对象的映射（dispatch 表）
tools = [stat_calc_tool, analyze_metric_tool, build_report_tool]
tool_map = {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# 第 2 步：读 Key + 建 LangChain 的 ChatModel
# ---------------------------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError(f"没读到 API Key！\n我找的文件是：{ENV_PATH}")

# ChatOpenAI 是 LangChain 对“聊天模型”的统一封装。
# 连百炼：api_key + base_url 换一下即可，和裸 openai 一样。
llm = ChatOpenAI(
    model="qwen3.8-flash",
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.0,
)

# bind_tools：把工具“挂”到模型上。之后模型如果觉得该调工具，
# 不会输出文字，而是返回一个结构化的 tool_calls 字段（这就是 Tool Calling）。
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 第 3 步：系统提示词（注意：不再需要 "Action: xxx" 格式指令）
# ---------------------------------------------------------------------------
# 因为模型走 tool_calling 协议，它自己知道“该调工具时返回 tool_calls”，
# 不用我们再教它输出文本格式。system 只需交代角色和规则。
SYSTEM_PROMPT = """你是一个行为数据分析 Agent。
根据用户问题，使用提供的工具分析人物A和人物B的行为数据，给出有数据支撑的回答。
所有数据必须通过工具获取，绝对不要凭空编造数字。
"""


# ---------------------------------------------------------------------------
# 第 4 步：ReAct 循环（Think → Act → Observe）—— 这次循环是“胶水代码”
# ---------------------------------------------------------------------------
# 闭环思想和前一版一样，但三处变了（这是 LangChain 版的核心）：
#   ① 工具用 @tool（自带 schema），不再是裸函数
#   ② 模型返回 tool_calls（结构化），不再用正则从文本抠 Action
#   ③ 工具结果用 ToolMessage 塞回（LangChain 消息类型，必须带 tool_call_id）
MAX_STEPS = 6


def run_agent(question: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for step in range(MAX_STEPS):
        print(f"\n{'=' * 60}\n[第 {step + 1} 轮 · Think]")
        ai_msg = llm_with_tools.invoke(messages)
        print("LLM:", ai_msg.content if ai_msg.content else "（决定调用工具，无文字）")

        # 把模型这条消息记进历史（含它的 tool_calls 意图）
        messages.append(ai_msg)

        # 有 tool_calls → 该动手了
        if ai_msg.tool_calls:
            for tc in ai_msg.tool_calls:
                name = tc["name"]
                args = tc["args"]
                print(f">>> [Act] 调用 {name}，参数={args}")
                # .invoke(args) 内部按 schema 把参数填进函数并执行
                result = tool_map[name].invoke(args)
                print(f">>> [Observe] {result}")
                # ToolMessage 必须带 tool_call_id，模型才知道“这是哪次调用的结果”
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            # 没 tool_calls → 模型认为可以直接回答了
            print(">>> 无工具调用，视为最终回答")
            return ai_msg.content

    return "（达到最大步数 MAX_STEPS，强制终止）"


# ---------------------------------------------------------------------------
# 第 5 步：主流程（三个问题，和模块1/前版完全一致）
# ---------------------------------------------------------------------------
def main():
    questions = [
        "在这次互动中，人物A的微笑次数是多少次？",
        "人物A和人物B相比，谁的说话时间更长？长多少秒？",
        "请分析一下这两个人的互动情况，生成一份简要报告。",
    ]
    for q in questions:
        print("\n\n##################################################")
        print("用户问题:", q)
        print("##################################################")
        final = run_agent(q)
        print("\n【Agent 最终回答】\n", final)


if __name__ == "__main__":
    main()
