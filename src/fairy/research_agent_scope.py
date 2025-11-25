
"""用户澄清与研究简报生成模块

本模块实现研究工作流的范围界定阶段，主要包括：
1. 评估用户请求是否需要澄清
2. 从对话中生成详细的研究简报

工作流使用结构化输出来做出确定性决策，
判断是否有足够的上下文信息来进行研究。
"""

from datetime import datetime
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from fairy.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from fairy.state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState

# ===== 工具函数 =====

def get_today_str() -> str:
    """获取当前日期的人类可读格式。"""
    return datetime.now().strftime("%a %b %-d, %Y")

# ===== 配置 =====

from fairy.init_model import init_model

# 初始化模型
model = init_model(model="gpt-4.1")

# ===== 工作流节点 =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    判断用户请求是否包含足够的信息以进行研究。

    使用结构化输出来做出确定性决策，避免产生幻觉。
    根据情况路由到研究简报生成节点，或以澄清问题结束。
    """
    # 设置结构化输出模型
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # 使用澄清指令调用模型
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])

    # 根据是否需要澄清进行路由
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    将对话历史转化为完整的研究简报。

    使用结构化输出确保简报遵循所需格式，
    并包含有效研究所需的全部必要细节。
    """
    # 设置结构化输出模型
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # 从对话历史生成研究简报
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # 更新状态中的研究简报并传递给上级代理
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

# ===== 图构建 =====

# 构建范围界定工作流
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# 添加工作流节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# 添加工作流边
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

# 编译工作流
scope_research = deep_researcher_builder.compile()
