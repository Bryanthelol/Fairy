
"""
研究代理的状态定义与 Pydantic 模式

本模块定义了研究代理工作流所使用的状态对象和结构化模式，
包括研究者状态管理和输出模式。
"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ===== 状态定义 =====

class ResearcherState(TypedDict):
    """
    研究代理的状态，包含消息历史和研究元数据。

    此状态跟踪研究者的对话、用于限制工具调用的迭代次数、
    正在研究的主题、压缩后的发现以及用于详细分析的原始研究笔记。
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    """
    研究代理的输出状态，包含最终研究结果。

    表示研究过程的最终输出，包括压缩后的研究发现
    以及研究过程中的所有原始笔记。
    """
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

# ===== 结构化输出模式 =====

class ClarifyWithUser(BaseModel):
    """范围界定阶段用户澄清决策的模式。"""
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
    """研究简报生成的模式。"""
    research_brief: str = Field(
        description="用于指导研究的研究问题。",
    )

class Summary(BaseModel):
    """网页内容摘要的模式。"""
    summary: str = Field(description="网页内容的简明摘要")
    key_excerpts: str = Field(description="内容中的重要引述和摘录")
