"""
One-time script: walk all 116 tasks, add appropriate `assert_*_policy_looked_up`
assertions to evaluation_criteria.env_assertions based on task category + keywords.

Policy dispatch rules:

  allergy             → server_food_safety, or ticket mentions allergy/allergic/peanut/gluten/dairy/seafood allergen
  incident_severity   → server_incident, server_celebration, spill/damage/harm tickets
  service_delay       → slow/wait/delay tickets
  reservation         → host_phone, host_seating, host_walkin
  promotion_stacking  → server_promotion where stacking/combining is at issue
  staff_authority     → escalation/authority/comp/discount > authority tickets, severe incidents
  membership          → membership/tier/points tickets (but only if not already about data lookup)

Usage: python scripts/add_policy_assertions.py
"""
import json
import re
from pathlib import Path

TASKS_PATH = Path(__file__).parent.parent / "hospitality_env" / "server" / "data" / "tasks.json"

# Keyword patterns per policy
KEYWORDS = {
    "allergy": [
        r"\ballerg", r"peanut", r"gluten", r"dairy", r"shellfish",
        r"lactose", r"cross[- ]contamin", r"vegan", r"vegetarian", r"kosher",
        r"halal", r"diabetic", r"low.sodium",
    ],
    "incident_severity": [
        r"spill", r"stain", r"damag", r"slip", r"fall", r"injur",
        r"harm", r"poisoning", r"burn", r"hurt", r"melted",
        r"ruined", r"broken", r"wrong soup", r"missing item", r"child",
        r"clothing", r"garment", r"cake wrong", r"destroyed",
    ],
    "service_delay": [
        r"\bslow\b", r"\bwait(ing|ed)?\b", r"\bdelay", r"taking too long",
        r"slow service", r"long wait", r"30\+ min",
    ],
    "reservation": [
        r"reservation", r"booking", r"\bno[- ]show\b", r"customer arriv",
        r"\blate\b", r"table avail", r"walk[- ]in", r"fully booked",
        r"squeeze", r"party size", r"holiday reservation",
    ],
    "promotion_stacking": [
        r"\bvoucher\b", r"coupon", r"discount", r"\bpromotion\b",
        r"lunch special", r"\bsms\b", r"stacking", r"\bcombine",
        r"redemption", r"secret code", r"birthday voucher",
        r"multiple discount",
    ],
    "staff_authority": [
        r"\bauthority\b", r"\bescalat", r"\bmanager\b", r"\bexceed",
        r"\b12%", r"\$10\b", r"compensation", r"comp item",
        r"round[- ]off", r"threat.*review",
    ],
    "membership": [
        r"member", r"tier", r"points", r"silver", r"gold", r"bronze",
        r"loyalty", r"signup",
    ],
}

# Category-level defaults (applied in addition to keyword scan)
CATEGORY_DEFAULTS = {
    "server_food_safety": ["allergy"],
    "server_incident": ["incident_severity"],
    "server_celebration": ["incident_severity"],  # celebrations are auto-severe context
    "host_phone": ["reservation"],
    "host_seating": ["reservation"],
    "host_walkin": ["reservation"],
    "server_promotion": ["promotion_stacking"],
    # server_billing intentionally not defaulted — varies from "what's the price"
    # (no policy) to "bill calculation error" (authority) to "post-payment coupon"
    # (promotion stacking). Let keywords decide.
}

ALL_POLICIES = list(KEYWORDS.keys())


def policies_for_task(task: dict) -> list[str]:
    """Return a list of policy names that should be looked up for this task."""
    policies = set()

    # Start with category defaults
    cat = task.get("description", {}).get("category", "")
    policies.update(CATEGORY_DEFAULTS.get(cat, []))

    # Scan text fields for keywords
    text_parts = [
        task.get("ticket", ""),
        task.get("description", {}).get("purpose", ""),
        task.get("description", {}).get("notes", ""),
    ]
    # relevant_policies can be a string or a list
    rp = task.get("description", {}).get("relevant_policies", "")
    if isinstance(rp, list):
        text_parts.extend(rp)
    else:
        text_parts.append(rp)
    text = " ".join(str(p) for p in text_parts).lower()

    for policy, patterns in KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text):
                policies.add(policy)
                break

    # Severe incidents should also check staff_authority (escalation required)
    case_level = task.get("description", {}).get("case_level", "").lower()
    if "severe" in case_level or "path d" in " ".join(str(p) for p in text_parts).lower():
        policies.add("staff_authority")
    if "incident_severity" in policies and case_level in ("moderate", "severe"):
        policies.add("staff_authority")

    # Cap at 3 per task to keep reward distribution balanced
    if len(policies) > 3:
        # Priority order when trimming
        priority = ["allergy", "incident_severity", "staff_authority",
                    "reservation", "promotion_stacking", "service_delay", "membership"]
        trimmed = [p for p in priority if p in policies][:3]
        policies = set(trimmed)

    return sorted(policies)


def add_assertions_to_task(task: dict) -> tuple[dict, list[str]]:
    """Add policy-lookup assertions to a task. Returns (modified_task, added_policies)."""
    policies = policies_for_task(task)
    if not policies:
        return task, []

    ec = task.setdefault("evaluation_criteria", {})
    assertions = ec.setdefault("env_assertions", [])

    # Check which assertions already exist (idempotent)
    existing_funcs = {a.get("func_name", "") for a in assertions}
    added = []
    for p in policies:
        func_name = f"assert_{p}_policy_looked_up"
        if func_name in existing_funcs:
            continue
        assertions.append({
            "env_type": "assistant",
            "func_name": func_name,
            "arguments": {},
            "assert_value": True,
            "message": f"Agent should query {p} policy before acting on this scenario",
        })
        added.append(p)

    return task, added


def main():
    with open(TASKS_PATH) as f:
        tasks = json.load(f)

    from collections import Counter
    policy_counts = Counter()
    task_coverage = 0
    tasks_unchanged = 0

    for t in tasks:
        _, added = add_assertions_to_task(t)
        if added:
            task_coverage += 1
            for p in added:
                policy_counts[p] += 1
        else:
            tasks_unchanged += 1

    with open(TASKS_PATH, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Total tasks: {len(tasks)}")
    print(f"Tasks with added policy assertions: {task_coverage}")
    print(f"Tasks unchanged (no relevant policy): {tasks_unchanged}")
    print(f"\nPolicy assertion distribution:")
    for policy, count in policy_counts.most_common():
        print(f"  {policy:22s} {count} tasks")
    print(f"\nTotal assertions added: {sum(policy_counts.values())}")


if __name__ == "__main__":
    main()
