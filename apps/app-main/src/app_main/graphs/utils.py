from typing import Any, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult

#: default_type van de chatgrafen naar de taak in het register. Transformatie is
#: een tekstbewerking en valt onder samenvatten; de rest is chat.
_DEFAULT_TYPE_NAAR_TAAK = {
    "chat": "chat",
    "large_context": "chat",
    "tools": "chat",
    "transformation": "summarization",
}


class PoortChatModel(BaseChatModel):
    """Een LangChain-chatmodel dat elke aanroep via ``ModelGate`` doet (F2-08).

    Alleen async (``ainvoke``): de chatgrafen roepen het zo aan. Elke aanroep is
    één regel in ``model_call``.
    """

    taak: str
    classificatie: Optional[str] = None
    run_id: Optional[str] = None
    max_tokens: Optional[int] = None
    json_mode: bool = False

    @property
    def _llm_type(self) -> str:
        return "modelpoort"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("PoortChatModel is alleen async; gebruik ainvoke")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from app_main.services.model_routing.llm_call import get_aanroeper

        antwoord = await get_aanroeper().chat(
            taak=self.taak, classificatie=self.classificatie, run_id=self.run_id,
            messages=convert_to_openai_messages(messages), json_mode=self.json_mode,
            max_tokens=self.max_tokens)
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=antwoord.tekst,
                              response_metadata={"model_name": antwoord.model}))])


async def provision_langchain_model(
    content, model_id, default_type, *, classificatie: Optional[str], **kwargs
) -> BaseChatModel:
    """Het chatmodel voor een stap van de chatgrafen, via de modelpoort (F2-08).

    Het model komt uit het register bij de taak (ADR 0007). ``model_id`` is een
    keuze uit de vroegere modelrijen in de database; die telt niet meer en
    vervalt in F2-09. ``classificatie`` is die van de invoer (verplicht);
    ``onbekend`` telt als intern.

    ``kwargs``: ``max_tokens`` en ``structured`` (JSON-antwoord).
    """
    return PoortChatModel(
        taak=_DEFAULT_TYPE_NAAR_TAAK.get(default_type, "chat"),
        classificatie=classificatie,
        max_tokens=kwargs.get("max_tokens"),
        json_mode=bool(kwargs.get("structured")),
    )
