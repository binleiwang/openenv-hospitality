"""
Shared agent utilities for hospitality_env.

This module contains the "agent wrapper" pieces that are model-agnostic:
- SYSTEM_PROMPT: role + behavior guidelines for whoever plays the server
- build_user_turn: render an env observation into a human-readable prompt
- parse_action: parse a model's JSON text into a HospitalityAction
- obs_to_dict: convert HospitalityObservation dataclass to dict for uniform access

Used by:
- baseline_eval.py       (Claude / Haiku / any API model)
- training_notebook.ipynb (Qwen via Unsloth, local GPU)
- future eval scripts     (any new model / checkpoint)

Keeping prompt + parser in one place guarantees all evaluations are comparable.
"""
import json
from hospitality_env import HospitalityAction


SYSTEM_PROMPT = """You are a customer service agent (server) at Berkeley Hot Pot restaurant.
You MUST follow the staff policy given in the task system_message.

At each turn you receive the customer's last message and any tool results.
You can respond with ONE of three actions, expressed as strict JSON:

  {"message": "<text to customer>", "tool_name": "", "tool_args": {}}
  {"message": "", "tool_name": "<tool>", "tool_args": {...}}
  {"message": "<text>", "tool_name": "<tool>", "tool_args": {...}}

Rules:
- Output ONLY the JSON object, no markdown fences, no commentary.
- Use tools before making factual claims (allergies, order details, policy limits).
- Escalate per policy when required (SEVERE incidents, authority limits).
- Keep messages concise and empathetic. Avoid small talk or filler.

Closing conversations (important — avoid runaway chats):
- Once you have provided an answer or solution, briefly confirm it meets the
  customer's need (e.g., "Does that resolve your concern?" or "Does this work
  for you?").
- If the customer seems satisfied, ask "Is there anything else I can help you
  with?" exactly once.
- If they say no (or express thanks/closure), send a warm, brief closing
  message and stop. Do NOT solicit new topics or extend the conversation.
- Goal: resolve the ticket efficiently, then close. A task completed in 4
  turns is better than the same task in 12 turns.
"""


def obs_to_dict(obs) -> dict:
    """Convert a HospitalityObservation dataclass to plain dict."""
    return {
        "customer_message": obs.customer_message,
        "tool_result": obs.tool_result,
        "tool_error": obs.tool_error,
        "system_message": obs.system_message,
        "available_tools": obs.available_tools,
        "tool_schemas": obs.tool_schemas,
        "task_description": obs.task_description,
        "turn_number": obs.turn_number,
        "max_turns": obs.max_turns,
        "metadata": obs.metadata,
    }


def build_user_turn(obs: dict, first_turn: bool) -> str:
    """Render an observation dict into the user message for the agent model."""
    parts = []
    if first_turn and obs.get("system_message"):
        parts.append(f"=== TASK CONTEXT ===\n{obs['system_message']}")
        if obs.get("tool_schemas"):
            parts.append(
                f"\n=== TOOL SCHEMAS ===\n"
                f"{json.dumps(obs['tool_schemas'], indent=2)}"
            )
    if obs.get("customer_message"):
        parts.append(f"\n=== CUSTOMER SAYS ===\n{obs['customer_message']}")
    if obs.get("tool_result"):
        parts.append(f"\n=== LAST TOOL RESULT ===\n{obs['tool_result']}")
    if obs.get("tool_error"):
        parts.append(f"\n=== LAST TOOL ERROR ===\n{obs['tool_error']}")
    parts.append(
        f"\n[turn {obs.get('turn_number', 0)}/{obs.get('max_turns', 20)}] "
        f"Respond with JSON."
    )
    return "\n".join(parts)


def parse_action(text: str) -> HospitalityAction:
    """Parse a model's JSON text response into a HospitalityAction (best effort).

    Handles:
    - Raw JSON objects
    - Markdown-fenced JSON (```json ... ```)
    - Trailing commentary (takes text between first { and last })
    - Completely non-JSON output (falls back to treating text as message)
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return HospitalityAction(message=text[:500])
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return HospitalityAction(message=text[:500])
    return HospitalityAction(
        message=data.get("message", "") or "",
        tool_name=data.get("tool_name", "") or "",
        tool_args=data.get("tool_args", {}) or {},
    )
