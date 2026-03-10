"""
huginn — Cognitive module for the Artux AI system.

Quick start:
    from huginn import build_huginn

    huginn = build_huginn(
        muninn      = MemoryAgent("artux.db"),
        llm_backend = "ollama",
        fast_model  = "qwen2.5:0.5b",
        sagax_model = "llama3.2",
        logos_model = "llama3.2:70b",
    )
    huginn.orchestrator.start(session=...)
"""

from .runtime.stm          import STMStore, STMEvent, ConsN
from .runtime.htm          import HTM, Task, ActiveSessionCache
from .runtime.perception   import PerceptionManager, ToolRegistry
from .runtime.orchestrator import Orchestrator, Session, ToolManager
from .agents.exilis        import Exilis
from .agents.sagax         import Sagax
from .agents.logos         import Logos
from .llm.client           import LLMClient, LLMPool

__all__ = [
    "STMStore", "STMEvent", "ConsN",
    "HTM", "Task", "ActiveSessionCache",
    "PerceptionManager", "ToolRegistry",
    "Orchestrator", "Session", "ToolManager",
    "Exilis", "Sagax", "Logos",
    "LLMClient", "LLMPool",
    "build_huginn",
]


def build_huginn(
    muninn,
    llm_backend:  str   = "ollama",
    fast_model:   str   = "qwen2.5:0.5b",
    sagax_model:  str   = "llama3.2",
    logos_model:  str   = "llama3.2",
    ollama_host:  str   = "http://localhost:11434",
    anthropic_key: str  = "",
    temperature:  float = 0.1,
    timeout_s:    float = 60.0,
    logos_interval_s: float = 300.0,
    on_tts_token: callable = None,
    on_ui_projection: callable = None,
) -> "HuginnInstance":
    """
    Factory: instantiate and wire all Huginn components.

    Returns a HuginnInstance with .orchestrator, .sagax, .logos, .stm, .htm
    all wired and ready. Call huginn.orchestrator.start() to begin.
    """

    def _llm(model):
        return LLMClient(
            backend     = llm_backend,
            model       = model,
            host        = ollama_host,
            api_key     = anthropic_key,
            temperature = temperature,
            timeout     = timeout_s,
        )

    stm   = STMStore(muninn)
    htm   = HTM()
    tools = ToolRegistry()

    session_ctx = Session()

    def _noop():
        pass

    perception = PerceptionManager(
        stm             = stm,
        htm             = htm,
        muninn          = muninn,
        tools           = tools,
        session         = session_ctx.as_context() if hasattr(session_ctx, "as_context")
                          else object(),
        on_event_written = _noop,   # re-wired after Exilis is created
        sig_threshold   = 0.88,
    )

    fast_llm  = _llm(fast_model)
    sagax_llm = _llm(sagax_model)
    logos_llm = _llm(logos_model)

    exilis = Exilis(
        stm            = stm,
        htm            = htm,
        llm            = fast_llm,
        on_act         = _noop,     # re-wired by Orchestrator.start()
        on_urgent      = _noop,
        poll_interval_s = 0.005,
    )

    sagax = Sagax(
        stm          = stm,
        htm          = htm,
        muninn       = muninn,
        llm          = sagax_llm,
        orchestrator = None,        # re-wired after Orchestrator is created
    )

    logos = Logos(
        stm        = stm,
        htm        = htm,
        muninn     = muninn,
        llm        = logos_llm,
        interval_s = logos_interval_s,
    )

    tool_manager = ToolManager(muninn)

    orchestrator = Orchestrator(
        stm                = stm,
        htm                = htm,
        perception         = perception,
        exilis             = exilis,
        sagax              = sagax,
        logos              = logos,
        tool_manager       = tool_manager,
        on_tts_token       = on_tts_token,
        on_ui_projection   = on_ui_projection,
    )

    # Wire back-references
    sagax.orchestrator = orchestrator
    perception.on_event_written = exilis.on_new_event

    return HuginnInstance(
        orchestrator = orchestrator,
        sagax        = sagax,
        logos        = logos,
        exilis       = exilis,
        stm          = stm,
        htm          = htm,
        tools        = tools,
        tool_manager = tool_manager,
    )


class HuginnInstance:
    """Container for all wired Huginn components."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def start(self, session: "Session" = None):
        self.orchestrator.start(session=session)

    def stop(self):
        self.orchestrator.stop()
