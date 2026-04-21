"""Print full scenario + evaluation criteria for a given task_id."""
import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python inspect_task.py <task_id>")
    sys.exit(1)

target = sys.argv[1]
p = Path(__file__).parent / "hospitality_env" / "server" / "data" / "tasks.json"
with open(p) as f:
    tasks = json.load(f)

t = next((x for x in tasks if x["id"] == target), None)
if t is None:
    print(f"Task '{target}' not found")
    sys.exit(1)

print(f"=== {t['id']} ===\n")
print(f"Ticket: {t.get('ticket', '')}\n")
desc = t.get("description", {})
print(f"Category     : {desc.get('category')}")
print(f"Purpose      : {desc.get('purpose')}")
print(f"Policies     : {desc.get('relevant_policies')}")
print(f"Case level   : {desc.get('case_level')}")
print(f"Notes        : {desc.get('notes', '')}")
us = t.get("user_scenario", {})
print(f"\n--- USER SCENARIO ---")
print(f"Persona      : {us.get('persona', '')}")
instr = us.get("instructions", {})
print(f"Reason       : {instr.get('reason_for_call', '')}")
print(f"Known info   : {instr.get('known_info', '')}")
print(f"Task instr   : {instr.get('task_instructions', '')}")
print(f"\n--- EVAL CRITERIA ---")
print(json.dumps(t.get("evaluation_criteria", {}), indent=2)[:2000])
