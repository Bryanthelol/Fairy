
"""研究范围界定的状态定义与 Pydantic 模式

本模块定义了研究代理范围界定工作流所使用的状态对象和结构化模式，
包括研究者状态管理和输出模式。
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ===== 状态定义 =====

class AgentInputState(MessagesState):
    """代理输入状态 - 仅包含来自用户输入的消息。"""
    pass

class AgentState(MessagesState):
    """
    多代理研究系统的主状态。

    扩展 MessagesState，添加用于研究协调的额外字段。
    注意：某些字段在不同状态类之间重复定义，以确保子图与主工作流之间
    状态管理的正确性。
    """

    # 从用户对话历史生成的研究简报
    research_brief: Optional[str]
    # 与上级代理交换的协调消息
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # 研究阶段收集的原始未处理笔记
    raw_notes: Annotated[list[str], operator.add] = []
    # 已处理和结构化的笔记，可用于报告生成
    notes: Annotated[list[str], operator.add] = []
    # 最终格式化的研究报告
    final_report: str

# ===== 结构化输出模式 =====

class ClarifyWithUser(BaseModel):
    """用户澄清决策和问题的模式。"""

    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题。",
    )
    question: str = Field(
        description="向用户提出的澄清报告范围的问题",
    )
    verification: str = Field(
        description="确认消息，表明在用户提供必要信息后将开始研究。",
    )

class ResearchQuestion(BaseModel):
    """结构化研究简报生成的模式。"""

    research_brief: str = Field(
        description="用于指导研究的研究问题。",
    )
