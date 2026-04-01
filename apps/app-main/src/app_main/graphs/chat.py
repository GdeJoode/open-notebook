import asyncio
from typing import Annotated, Optional

from ai_prompter import Prompter
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from shared.models import Notebook
from app_main.config import LANGGRAPH_CHECKPOINT_FILE
from app_main.graphs.utils import provision_langchain_model

MODEL_INVOKE_TIMEOUT = 300  # seconds


class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]
    notebook: Optional[Notebook]
    context: Optional[str]
    context_config: Optional[dict]
    model_override: Optional[str]


async def call_model_with_messages(state: ThreadState, config: RunnableConfig) -> dict:
    system_prompt = Prompter(prompt_template="chat").render(data=state)
    payload = [SystemMessage(content=system_prompt)] + state.get("messages", [])
    model_id = config.get("configurable", {}).get("model_id") or state.get(
        "model_override"
    )

    model = await provision_langchain_model(
        str(payload), model_id, "chat", max_tokens=8192
    )

    ai_message = await asyncio.wait_for(
        model.ainvoke(payload), timeout=MODEL_INVOKE_TIMEOUT
    )
    return {"messages": ai_message}


memory = SqliteSaver.from_conn_string(LANGGRAPH_CHECKPOINT_FILE)

agent_state = StateGraph(ThreadState)
agent_state.add_node("agent", call_model_with_messages)
agent_state.add_edge(START, "agent")
agent_state.add_edge("agent", END)
graph = agent_state.compile(checkpointer=memory)
