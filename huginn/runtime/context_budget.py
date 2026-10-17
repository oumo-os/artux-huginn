"""
runtime/context_budget.py — shared context-budget helpers for Sagax.

Capability surfaces (ACTION TAGS) are the only unbounded context layer in
the system — every unique permission_scope of installed + hot tools grows
with the arsenal. These helpers guarantee the surface always fits the
small-model context window:

  - fit_user_prompt(): post-assembly guard for the formatted user prompt.
    If over budget, the lossy narrative layers are stripped; capability
    surfaces are never cut by this path.

Both are state-tunable from sagax.* states at the call site.
"""

import re


LOSSY_LAYER_KEYS = (
    "live_context_snippet",
    "cons_n_snippet",
    "recent_events",
    "state_snapshot",
    "active_tasks",
    "staging_tools",
    "injected_instruction",
)

_FMT_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def fit_user_prompt(
    template: str,
    fmt_kwargs: dict,
    budget_chars: int,
    mode: str = "",
    emit_truncated=None,
) -> str:
    """
    Format a user prompt from `template` + `fmt_kwargs`, then apply a
    post-assembly fit guard.

    If the result is under budget, it is returned unchanged. If over,
    the prompt is re-formatted with the lossy narrative layers stripped
    (capability surfaces — action tags, grants, entity block — are never
    cut). Returns the final prompt string.

    `emit_truncated` is an optional callable(mode, bytes_saved, layer)
    for observability (Sagax records a context_truncated STM event).
    """
    user_prompt = template.format(**fmt_kwargs)
    if len(user_prompt) <= budget_chars:
        return user_prompt

    trimmed = {k: v for k, v in fmt_kwargs.items() if k not in LOSSY_LAYER_KEYS}
    trimmed["cons_n_snippet"] = "(context trimmed to fit budget)"

    # Drop placeholders that reference stripped layers instead of
    # letting format() raise KeyError on templates that use them.
    rebuilt_template = template
    for key in _FMT_PLACEHOLDER.findall(template):
        if key not in trimmed:
            rebuilt_template = rebuilt_template.replace("{" + key + "}", "")

    rebuilt = rebuilt_template.format(**trimmed)
    if callable(emit_truncated):
        try:
            emit_truncated(
                mode,
                max(0, len(user_prompt) - len(rebuilt)),
                "narrative/events/state",
            )
        except Exception:
            pass
    return rebuilt
