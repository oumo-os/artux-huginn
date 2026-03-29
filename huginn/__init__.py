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
from .runtime.actuation_bus import ActuationBus
from .runtime.actuation_manager import ActuationManager
from .runtime.htm          import HTM, Task, ActiveSessionCache
from .runtime.perception   import PerceptionManager, ToolRegistry, SessionContext
from .runtime.orchestrator import Orchestrator, Session
from .runtime.tool_manager import ToolManager, ToolDescriptor, register_tool
from .agents.exilis        import Exilis
from .agents.sagax         import Sagax
from .agents.logos         import Logos
from .llm.client           import LLMClient, LLMPool

__all__ = [
    "STMStore", "STMEvent", "ConsN",
    "HTM", "Task", "ActiveSessionCache",
    "PerceptionManager", "ToolRegistry",
    "Orchestrator", "Session",
    "ToolManager", "ToolDescriptor", "register_tool",
    "Exilis", "Sagax", "Logos",
    "LLMClient", "LLMPool",
    "ActuationBus", "ActuationManager",
    "build_huginn",
]



def _assign_gguf_models(models_dir, htm) -> dict:
    """
    Scan the models directory and assign GGUF files to agent roles.

    Each role benefits from a different model size:
      exilis → tiny/small (0.5B–1.5B): single JSON triage call, temperature 0
      sagax  → medium (3B–8B): full reasoning and Narrator stream
      logos  → largest available (7B–13B+): deep consolidation and synthesis

    Assignment priority (each role stops at first match):
      1. HTM.states already set   — LTM recall or prior state_set wins
      2. models/models.yaml       — explicit per-role filename config
      3. Prefix-named .gguf files — exilis_*.gguf, sagax_*.gguf, logos_*.gguf
      4. Size-ordered fallback    — smallest→exilis, middle→sagax, largest→logos
      5. Single file              — shared across all roles (dev/test)

    models.yaml example:
      exilis: "qwen2.5-0.5b-instruct-q8_0.gguf"
      sagax:  "llama-3.2-3b-instruct-q4_k_m.gguf"
      logos:  "llama-3.1-8b-instruct-q4_k_m.gguf"
    """
    import pathlib as _pl
    _models_dir = _pl.Path(models_dir)
    PROVIDER = "tool.llm.llamacpp.v1"
    assigned: dict = {}

    def _already_set(role):
        return bool(htm.states.get(f"{role}.provider"))

    def _set_role(role, model_path):
        if _already_set(role):
            return
        # Per-role model path key so llamacpp tool can select the right file
        htm.states.set(f"tool.llm.llamacpp.v1.model_path.{role}", model_path, mark_dirty=False)
        htm.states.set(f"{role}.provider", PROVIDER, mark_dirty=False)
        assigned[role] = model_path

    if not _models_dir.exists():
        return {}

    all_gguf = sorted(_models_dir.glob("*.gguf"))

    # Priority 1: models.yaml
    yaml_path = _models_dir / "models.yaml"
    if yaml_path.exists():
        data = {}
        try:
            import yaml as _y
            data = _y.safe_load(yaml_path.read_text()) or {}
        except ImportError:
            for line in yaml_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and ":" in line:
                    k, _, v = line.partition(":")
                    data[k.strip()] = v.strip().strip('"').strip("'")
        for role in ("exilis", "sagax", "logos"):
            fname = data.get(role, "")
            if fname:
                fp = _pl.Path(fname)
                if not fp.is_absolute():
                    fp = _models_dir / fname
                if fp.exists():
                    _set_role(role, str(fp))
        if assigned:
            return assigned

    # Priority 2: prefix-named files
    PREFIX_MAP = {
        "exilis": "exilis", "small": "exilis", "fast": "exilis",
        "sagax":  "sagax",  "medium": "sagax",
        "logos":  "logos",  "large": "logos",
    }
    for f in all_gguf:
        stem = f.stem.lower()
        for pfx, role in PREFIX_MAP.items():
            if stem.startswith(pfx + "_") or stem.startswith(pfx + "-"):
                _set_role(role, str(f))
                break
    if assigned:
        return assigned

    # Priority 3: size-ordered fallback
    if not all_gguf:
        return {}

    sized = sorted(all_gguf, key=lambda f: f.stat().st_size)
    if len(sized) == 1:
        for role in ("exilis", "sagax", "logos"):
            _set_role(role, str(sized[0]))
    elif len(sized) == 2:
        _set_role("exilis", str(sized[0]))
        _set_role("sagax",  str(sized[1]))
        _set_role("logos",  str(sized[1]))
    else:
        for role, f in zip(("exilis", "sagax", "logos"), sized):
            _set_role(role, str(f))

    return assigned

def build_huginn(
    muninn,
    fallback_model:   str   = "llama3.2",
    fallback_backend: str   = "ollama",
    fallback_host:    str   = "http://localhost:11434",
    logos_interval_s: float = 300.0,
    on_tts_token:     callable = None,
    on_ui_projection: callable = None,
    staging_dir:      str   = "",
    active_dir:       str   = "",
) -> "HuginnInstance":
    """
    Factory: instantiate and wire all Huginn components.

    LLM configuration (backend, model names, API keys, host) is recalled
    from Muninn LTM at startup — stored as class_type="config" entries
    with distinctive topic keys (e.g. "artux.config.llm.v1").

    The fallback_* parameters are only used when no config exists in LTM
    (first-ever boot, before Logos has written defaults). After the first
    Logos pass, recalled config takes over.

    Parameters
    ----------
    muninn : MemoryAgent
        The Muninn memory backend. All config and knowledge lives here.
    fallback_model : str
        Model name used for all agents when no LTM config is found.
    fallback_backend : str
        "ollama" | "anthropic" — used when no LTM config is found.
    fallback_host : str
        Ollama host URL used when no LTM config is found.
    staging_dir : str
        Directory Logos watches for new tool .py files.
        Defaults to ./tools/staging relative to the DB file location.
    active_dir : str
        Directory installed tool files are moved into.
        Defaults to ./tools/active relative to the DB file location.

    Returns a HuginnInstance. Call huginn.start() to begin.
    """
    from .runtime.tool_discovery import ToolDiscovery
    import os
    import pathlib

    # Resolve staging/active dirs relative to DB path if not given
    db_path = (
        getattr(getattr(muninn, "db", None), "path", None)
        or getattr(getattr(muninn, "db", None), "db_path", None)
        or "artux.db"
    )
    base = os.path.dirname(os.path.abspath(db_path))
    if not staging_dir:
        staging_dir = os.path.join(base, "tools", "staging")
    if not active_dir:
        active_dir = os.path.join(base, "tools", "active")

    stm          = STMStore(muninn)
    htm          = HTM()

    # ── GGUF model auto-detection ──────────────────────────────────────
    # Each cognitive role needs a different model size for efficiency:
    #   Exilis/consN → tiny/small (0.5B–1.5B): one JSON call, temperature 0
    #   Sagax        → medium (3B–8B): full reasoning and Narrator stream
    #   Logos        → largest available (7B–13B+): deep consolidation
    #
    # Assignment priority (each role stops at first match):
    #   1. HTM.states already set (LTM recall or prior state_set)
    #   2. models/models.yaml  — explicit per-role config
    #   3. Prefix-named files  — exilis_*.gguf, sagax_*.gguf, logos_*.gguf
    #   4. Single fallback     — any .gguf, shared across all unassigned roles
    #   5. _BuiltinProvider    — Ollama/Anthropic from LTM config
    _gguf_assigned = _assign_gguf_models(
        pathlib.Path(db_path).parent / "models", htm
    )
    # ─────────────────────────────────────────────────────────────────

    pipeline_tools  = ToolRegistry()
    tool_manager    = ToolManager(muninn, htm)
    actuation_bus   = ActuationBus()
    actuation_mgr   = ActuationManager(bus=actuation_bus, htm=htm, stm=stm)

    def _llm(role: str) -> LLMClient:
        """
        Create a role-aware LLMClient.

        The client holds a reference to HTM.states, which is populated by
        Orchestrator._apply_system_config() immediately after start().
        Every call reads the live provider/model/temperature from states,
        so a Sagax state_set takes effect on the next inference call.

        Fallback values keep the system working before config is recalled.
        """
        return LLMClient(
            role             = role,
            htm              = htm,
            fallback_backend = fallback_backend,
            fallback_model   = fallback_model,
            fallback_host    = fallback_host,
            fallback_temp    = 0.1,
            fallback_timeout = 60.0,
        )

    _noop = lambda: None

    perception = PerceptionManager(
        stm              = stm,
        htm              = htm,
        muninn           = muninn,
        tools            = pipeline_tools,
        session          = SessionContext(),
        on_event_written = _noop,            # re-wired after Exilis is created
        sig_threshold    = 0.88,
    )

    # All three agents share the same fallback LLM config at boot.
    # Orchestrator.start() recalls artux.config.llm.* from Muninn and
    # calls reconfigure() on each client with the appropriate role config.
    fast_llm  = _llm("exilis")
    sagax_llm = _llm("sagax")
    logos_llm = _llm("logos")

    exilis = Exilis(
        stm             = stm,
        htm             = htm,
        llm             = fast_llm,
        on_act          = _noop,             # re-wired by Orchestrator.start()
        on_urgent       = _noop,
        idle_yield_s    = 0.0,
    )

    sagax = Sagax(
        stm          = stm,
        htm          = htm,
        muninn       = muninn,
        llm          = sagax_llm,
        orchestrator = None,                 # re-wired after Orchestrator is created
    )

    logos = Logos(
        stm          = stm,
        htm          = htm,
        muninn       = muninn,
        llm          = logos_llm,
        tool_manager = tool_manager,
        discovery    = ToolDiscovery(
            staging_dir = staging_dir,
            active_dir  = active_dir,
            stm         = stm,
            htm         = htm,
        ),
        interval_s   = logos_interval_s,
    )

    orchestrator = Orchestrator(
        stm               = stm,
        htm               = htm,
        perception        = perception,
        exilis            = exilis,
        sagax             = sagax,
        logos             = logos,
        tool_manager      = tool_manager,
        on_tts_token      = on_tts_token,
        on_ui_projection  = on_ui_projection,
        actuation_bus     = actuation_bus,
        actuation_manager = actuation_mgr,
    )

    # Wire back-references
    sagax.orchestrator           = orchestrator
    perception.on_event_written  = exilis.on_new_event

    return HuginnInstance(
        orchestrator      = orchestrator,
        sagax             = sagax,
        logos             = logos,
        exilis            = exilis,
        perception        = perception,
        stm               = stm,
        htm               = htm,
        tools             = pipeline_tools,
        tool_manager      = tool_manager,
        actuation_bus     = actuation_bus,
        actuation_manager = actuation_mgr,
    )


class HuginnInstance:
    """Container for all wired Huginn components."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def start(self, session: "Session" = None):
        self.orchestrator.start(session=session)
        # Execute the startup procedure after all components are running.
        # Sagax recalls procedure.startup.v1 from LTM and emits boot tokens.
        import threading
        threading.Thread(
            target=self.sagax.execute_startup_procedure,
            daemon=True,
            name="SagaxBoot",
        ).start()

    def stop(self):
        self.orchestrator.stop()
