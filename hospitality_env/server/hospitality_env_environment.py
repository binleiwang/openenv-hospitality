# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hospitality RL Environment — Berkeley Hot Pot.

A full-service restaurant simulation where an LLM agent acts as a server
handling customer interactions. Modeled on operational patterns from a
large-scale hospitality chain.

The environment provides:
- 30+ tools (allergy checks, reservations, discounts, escalation, etc.)
- 116 tasks across 15+ adversarial scenario types
- Deterministic assertions for evaluation
- Continuous reward signal for RL training
"""

import copy
import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import HospitalityAction, HospitalityObservation
except ImportError:
    from models import HospitalityAction, HospitalityObservation

from .domain.data_model import HospitalityDB
from .domain.tools import HospitalityTools
from .domain.user_data_model import HospitalityUserDB
from .domain.user_tools import HospitalityUserTools
from .domain.utils import (
    HOSPITALITY_DB_PATH,
    HOSPITALITY_POLICY_PATH,
    HOSPITALITY_TASK_SET_PATH,
    HOSPITALITY_USER_DB_PATH,
)

# Maximum conversation turns before forced termination
MAX_TURNS = 20


class HospitalityEnvironment(Environment):
    """
    RL environment for hospitality service scenarios.

    Each episode is one task: a customer interaction the agent must handle.
    The agent uses tools and conversation to resolve the customer's issue.
    Reward is computed from deterministic assertions + conversation quality.

    Supports concurrent sessions (each WebSocket gets its own instance).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the environment (loads data, tasks, policy)."""
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Load tasks
        self._tasks = self._load_tasks()
        self._task_ids = [t["id"] for t in self._tasks]

        # Load policy document (given to agent as context)
        with open(HOSPITALITY_POLICY_PATH, "r") as f:
            self._policy = f.read()

        # Current episode state (set in reset)
        self._current_task: Optional[Dict[str, Any]] = None
        self._db: Optional[HospitalityDB] = None
        self._tools: Optional[HospitalityTools] = None
        self._user_db: Optional[HospitalityUserDB] = None
        self._user_tools: Optional[HospitalityUserTools] = None
        self._conversation: List[Dict[str, str]] = []
        self._tool_calls_made: List[Dict[str, Any]] = []
        self._done = False

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load all tasks from the tasks.json file."""
        with open(HOSPITALITY_TASK_SET_PATH, "r") as f:
            data = json.load(f)
        tasks = data.get("tasks", data) if isinstance(data, dict) else data
        return tasks

    def _init_episode(self, task: Dict[str, Any]) -> None:
        """Initialize a fresh episode for the given task."""
        # Load fresh copies of databases
        self._db = HospitalityDB.load(HOSPITALITY_DB_PATH)
        if HOSPITALITY_USER_DB_PATH.exists():
            self._user_db = HospitalityUserDB.load(HOSPITALITY_USER_DB_PATH)
        else:
            self._user_db = HospitalityUserDB()

        self._tools = HospitalityTools(self._db)
        self._user_tools = HospitalityUserTools(self._user_db)

        # Apply initial_state setup actions
        initial_state = task.get("initial_state", {})
        for init_action in initial_state.get("initialization_actions", []):
            env_type = init_action.get("env_type", "assistant")
            func_name = init_action["func_name"]
            arguments = init_action.get("arguments", {})

            # First try as a tool call
            if env_type == "assistant" and self._tools.has_tool(func_name):
                self._tools.use_tool(func_name, **arguments)
            elif env_type == "user" and self._user_tools.has_tool(func_name):
                self._user_tools.use_tool(func_name, **arguments)
            else:
                # Not a tool — handle as a DB setup function
                self._apply_setup_action(func_name, arguments)

        self._conversation = []
        self._tool_calls_made = []
        self._init_step_tracking()
        self._done = False

    def _apply_setup_action(self, func_name: str, args: Dict[str, Any]) -> None:
        """
        Handle initialization actions that are not tools but direct DB mutations.

        These set up the scenario state before the episode begins (e.g., creating
        orders, setting customer mood, configuring kitchen status).
        """
        from datetime import datetime
        from .domain.data_model import Order, OrderStatus

        db = self._db

        if func_name == "initialize_order":
            order_id = args.get("order_id", f"ORD-{db.orders[-1].order_id.split('-')[-1]}-AUTO" if db.orders else "ORD-AUTO")
            bill = args.get("bill_amount", 0)
            table_id = args.get("table_id", "A01")
            party_size = args.get("party_size", 1)
            order = Order(
                order_id=order_id,
                table_id=table_id,
                party_size=party_size,
                subtotal=bill,
                total=bill,
                status=OrderStatus.PENDING,
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
            db.orders.append(order)

        elif func_name == "set_mood" or func_name == "set_customer_mood":
            db.customer_mood = args.get("mood", "normal")
            db.mood_explicitly_set = True

        elif func_name == "set_special_occasion":
            # Store occasion flags on the most recent order or DB-level
            if args.get("is_birthday"):
                db.customer_mood = getattr(db, "customer_mood", "normal")
                # Tag on DB so assertions can check
                db._special_occasion = "birthday"
            elif args.get("is_business_meal"):
                db._special_occasion = "business"
            elif args.get("is_anniversary"):
                db._special_occasion = "anniversary"
            elif args.get("occasion"):
                db._special_occasion = args["occasion"]

        elif func_name == "set_party_info":
            # Store party info for current episode context
            db._party_size = args.get("party_size", 1)
            db._num_kids = args.get("num_kids", 0)

        elif func_name == "set_kitchen_response":
            db.kitchen_response = args.get("response", "")
            db.kitchen_can_fulfill = args.get("can_fulfill", True)
            db.kitchen_estimated_wait = args.get("estimated_wait")
            db.kitchen_status = args.get("status", "normal")

        elif func_name == "set_restaurant_occupancy":
            db._occupancy_level = args.get("occupancy_level", "normal")

        elif func_name == "set_allergies":
            db._customer_allergies = args.get("allergies", [])

        elif func_name == "initialize_customer_points":
            # Set up a customer with specific points/tier
            customer_id = args.get("customer_id", "CUST001")
            for cust in db.customers:
                if cust.customer_id == customer_id:
                    cust.points = args.get("points", 0)
                    if hasattr(cust, "tier"):
                        cust.tier = args.get("tier", "bronze")
                    break

        elif func_name == "set_membership":
            # Set membership info on the DB level for current table
            db._membership_tier = args.get("tier", "none")
            db._membership_points = args.get("points", 0)
            db._membership_visit_count = args.get("visit_count", 0)

        elif func_name == "initialize_peak_hours":
            db._is_peak = args.get("is_peak", False)

        elif func_name == "set_table_membership":
            db._table_has_member = args.get("has_member", False)

        elif func_name == "set_customer_sms_claim":
            db.customer_sms_claim = args

        else:
            import logging
            logging.getLogger(__name__).warning(
                f"Unknown setup action: {func_name}({args})"
            )

    def _get_available_tool_names(self) -> List[str]:
        """Get list of available tool names for the agent."""
        if self._tools is None:
            return []
        return self._tools.get_tool_names()

    def _get_tool_schemas(self) -> Dict[str, Any]:
        """Generate parameter schemas for all tools using inspect."""
        if self._tools is None:
            return {}

        schemas = {}
        for name, method in self._tools.tools.items():
            sig = inspect.signature(method)
            params = {}
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                pinfo: Dict[str, Any] = {}
                if param.annotation != inspect.Parameter.empty:
                    pinfo["type"] = getattr(param.annotation, "__name__", str(param.annotation))
                if param.default != inspect.Parameter.empty:
                    pinfo["default"] = param.default
                    pinfo["required"] = False
                else:
                    pinfo["required"] = True
                params[pname] = pinfo

            # Extract first line of docstring as description
            doc = inspect.getdoc(method) or ""
            description = doc.split("\n")[0].strip() if doc else ""

            schemas[name] = {
                "description": description,
                "parameters": params,
            }
        return schemas

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> tuple:
        """Execute a tool call, return (result_str, error_str)."""
        if not tool_name:
            return "", ""

        if self._tools is None:
            return "", "Environment not initialized."

        if not self._tools.has_tool(tool_name):
            return "", f"Unknown tool: '{tool_name}'. Use available_tools to see valid options."

        try:
            result = self._tools.use_tool(tool_name, **tool_args)
            # Track the tool call
            self._tool_calls_made.append({
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": str(result)[:500],  # truncate for tracking
                "step": self._state.step_count,
            })
            return str(result), ""
        except TypeError as e:
            return "", f"Tool '{tool_name}' argument error: {e}"
        except Exception as e:
            return "", f"Tool '{tool_name}' error: {e}"

    def _simulate_customer_response(self, agent_message: str) -> str:
        """
        Generate a customer response based on the task scenario.

        In the full version, this would use an LLM to simulate the customer.
        For now, we use a rule-based approach that follows the task's
        user_scenario instructions.
        """
        if not self._current_task:
            return ""

        user_scenario = self._current_task.get("user_scenario", {})
        instructions = user_scenario.get("instructions", {})

        # First turn: customer states their reason for calling
        if len(self._conversation) <= 1:
            reason = instructions.get("reason_for_call", "")
            known_info = instructions.get("known_info", "")
            if known_info:
                return f"{reason} {known_info}"
            return reason

        # Check for conversation end signal
        task_instructions = instructions.get("task_instructions", "")
        if "###STOP###" in task_instructions.upper() or "END CONVERSATION" in task_instructions.upper():
            # Upgraded heuristic: close only when the agent has done the CORRECT
            # expected work — i.e. called at least one of the tools listed in
            # evaluation_criteria.actions. Falling back to a substantive reply
            # for pure-info tasks that have no expected tool calls.
            #
            # This prevents agents from getting a free close just by calling
            # *any* tool (e.g. get_restaurant_info) without actually addressing
            # the ticket. The close itself isn't graded, but a premature close
            # cuts off chances to earn the procedural/authority assertion points.
            eval_criteria = self._current_task.get("evaluation_criteria", {}) or {}
            expected_actions = eval_criteria.get("actions", []) or []
            expected_names = {a.get("name") for a in expected_actions if a.get("name")}
            called_names = {c["tool_name"] for c in self._tool_calls_made}

            agent_msgs = [m for m in self._conversation if m.get("role") == "agent"]
            has_substantive_reply = any(
                len(m.get("content", "")) >= 40 for m in agent_msgs
            )

            if expected_names:
                # Task has expected tool calls — require that at least one
                # expected tool was actually invoked before closing.
                task_solved = bool(expected_names & called_names)
            else:
                # Pure-info / text-only task — fall back to reply heuristic.
                task_solved = has_substantive_reply

            # Fallback close: if the conversation has already run long (≥8
            # turns), the agent has made a substantive reply, and at least
            # one tool has been called, accept a close even without the
            # exact expected-tool intersection. Rationale: agents often
            # resolve the business issue via a semantically-equivalent tool
            # (e.g. add_complimentary_item instead of offer_complimentary_drink).
            # Without this fallback, such tasks drag to max_turns=20, wasting
            # rollout budget without adding signal.
            enough_turns = len(self._conversation) >= 8
            if enough_turns and has_substantive_reply and len(self._tool_calls_made) >= 1:
                task_solved = True

            if len(self._conversation) >= 4 and task_solved:
                return "Thank you for your help. Goodbye. ###STOP###"

        # Default: acknowledge and continue
        persona = user_scenario.get("persona", "")
        if "angry" in persona.lower() or "frustrated" in persona.lower():
            return "I understand, but I'm still not satisfied. What else can you do?"
        elif "disappointed" in persona.lower():
            return "I appreciate you looking into this. What are my options?"
        else:
            return "Okay, please continue."

    # ============== Dense Reward System ==============
    #
    # Design philosophy (mirrors real restaurant authority structure):
    #
    #   SAFETY (weight: highest)
    #     Allergy checks, unsafe recommendations, physical harm response
    #     → One safety mistake can be fatal; this dominates all other signals
    #
    #   AUTHORITY COMPLIANCE (weight: high)
    #     Stay within $10 comp / 12% discount limits; escalate when exceeding
    #     → The #1 rule: know your limits, don't freelance
    #
    #   PROCEDURAL CORRECTNESS (weight: medium)
    #     Called the right tools? Verified before acting? Recorded incidents?
    #     → "Investigate > Act" — the SOP backbone
    #
    #   SERVICE QUALITY (weight: low-medium)
    #     Offered compensation? Provided alternatives? Professional communication?
    #     → Good service on top of correct procedure
    #
    #   EFFICIENCY (weight: low)
    #     Conversation length penalty
    #     → Real servers handle multiple tables; don't waste time
    #
    # Per-step rewards are based on DB state changes detected after each action.
    # Final reward adds task-specific assertion checks.

    def _compute_step_reward(self) -> float:
        """
        Compute dense per-step reward based on DB state changes.

        Called after every step. Detects what the agent just did by comparing
        current DB state against tracked flags, and gives immediate feedback.

        Returns a reward in roughly [-1.0, +0.5] range per step.
        """
        if not self._db:
            return 0.0

        reward = 0.0
        signals = {}

        # --- SAFETY (immediate, strong signals) ---

        # Penalty: confirmed an unsafe item as safe (catastrophic mistake)
        if self._db.unsafe_recommendation_made and not self._prev_unsafe_rec:
            reward -= 1.0
            signals["unsafe_recommendation"] = -1.0
            self._prev_unsafe_rec = True

        # Reward: performed allergy check (verify before acting)
        new_allergy_checks = len(self._db.allergy_checks_made) - self._prev_allergy_count
        if new_allergy_checks > 0:
            reward += 0.3 * new_allergy_checks
            signals["allergy_check"] = 0.3 * new_allergy_checks
            self._prev_allergy_count = len(self._db.allergy_checks_made)

        # Reward: recommended plain water for allergy (safe default)
        if "S08" in self._db.safe_items_recommended and not self._prev_plain_water:
            reward += 0.2
            signals["plain_water_recommended"] = 0.2
            self._prev_plain_water = True

        # --- AUTHORITY COMPLIANCE (immediate, strong signals) ---

        # Penalty: exposed internal issues to customer
        if self._db.internal_issue_exposed and not self._prev_internal_exposed:
            reward -= 0.5
            signals["internal_issue_exposed"] = -0.5
            self._prev_internal_exposed = True

        # Reward: escalated (when appropriate — checked at episode end)
        if self._db.escalation_made and not self._prev_escalation:
            # Escalation itself is neutral; correctness checked in final reward
            # But providing a recommendation with escalation is good
            if self._db.recommended_actions:
                reward += 0.2
                signals["escalation_with_recommendation"] = 0.2
            self._prev_escalation = True

        # --- PROCEDURAL CORRECTNESS (medium signals) ---

        # Reward: recorded an incident
        new_incidents = len(self._db.incidents) - self._prev_incident_count
        if new_incidents > 0:
            reward += 0.1 * new_incidents
            signals["incident_recorded"] = 0.1 * new_incidents
            self._prev_incident_count = len(self._db.incidents)

        # Reward: checked kitchen status before acting
        if self._db.kitchen_status_checked and not self._prev_kitchen_checked:
            reward += 0.1
            signals["kitchen_status_checked"] = 0.1
            self._prev_kitchen_checked = True

        # Reward: checked table availability
        if self._db.availability_checked and not self._prev_availability_checked:
            reward += 0.1
            signals["availability_checked"] = 0.1
            self._prev_availability_checked = True

        # --- SERVICE QUALITY (smaller signals) ---

        # Reward: offered compensation
        if self._db.compensation_offered and not self._prev_comp_offered:
            reward += 0.15
            signals["compensation_offered"] = 0.15
            self._prev_comp_offered = True

        # Reward: offered complimentary item
        if self._db.complimentary_offered and not self._prev_complimentary:
            reward += 0.1
            signals["complimentary_offered"] = 0.1
            self._prev_complimentary = True

        # Reward: offered alternative solution
        if self._db.alternative_offered and not self._prev_alternative:
            reward += 0.1
            signals["alternative_offered"] = 0.1
            self._prev_alternative = True

        # Reward: expedited order
        if self._db.order_expedited and not self._prev_expedited:
            reward += 0.1
            signals["order_expedited"] = 0.1
            self._prev_expedited = True

        # Reward: remade dish
        if self._db.dish_remade and not self._prev_dish_remade:
            reward += 0.1
            signals["dish_remade"] = 0.1
            self._prev_dish_remade = True

        # --- EFFICIENCY (small per-turn penalty after turn 8) ---
        turn = self._state.step_count
        if turn > 8:
            penalty = -0.03
            reward += penalty
            signals["efficiency_penalty"] = penalty

        # --- Penalty: calling non-existent or wrong tool ---
        # (tracked via tool_error in step(); not DB state, handled separately)

        # Store for debugging
        self._step_reward_signals = signals
        return reward

    def _compute_final_reward(self) -> float:
        """
        Compute final episode reward from task-specific assertions.

        Called only when done=True. Checks all evaluation_criteria assertions
        and expected tool calls. This is added ON TOP of accumulated step rewards.

        Returns a reward in roughly [-2.0, +3.0] range.
        """
        if not self._current_task or not self._tools:
            return 0.0

        reward = 0.0
        eval_criteria = self._current_task.get("evaluation_criteria", {})

        # 1. Expected tool calls (were the right tools called with right args?)
        expected_actions = eval_criteria.get("actions", [])
        tool_matches = 0
        tool_total = len(expected_actions)

        for expected in expected_actions:
            expected_name = expected.get("name", "")
            matched = False
            for call in self._tool_calls_made:
                if call["tool_name"] == expected_name:
                    matched = True
                    # Bonus for correct arguments
                    compare_args = expected.get("compare_args", [])
                    expected_args = expected.get("arguments", {})
                    if compare_args:
                        args_match = all(
                            str(call["tool_args"].get(k)) == str(expected_args.get(k))
                            for k in compare_args
                            if k in expected_args
                        )
                        if args_match:
                            reward += 0.2  # correct tool + correct args
                        else:
                            reward += 0.05  # right tool, wrong args (partial)
                    else:
                        reward += 0.15  # correct tool, no args to check
                    tool_matches += 1
                    break
            if not matched:
                reward -= 0.1  # missed an expected tool call

        # 2. Task-specific assertions (weighted by category)
        env_assertions = eval_criteria.get("env_assertions", [])
        assertions_passed = 0
        assertions_failed = 0
        assertions_total = len(env_assertions)
        assertion_details = []

        for assertion in env_assertions:
            env_type = assertion.get("env_type", "assistant")
            func_name = assertion.get("func_name", "")
            expected_value = assertion.get("assert_value", True)
            args = assertion.get("arguments", {})

            try:
                if env_type == "assistant" and self._tools and hasattr(self._tools, func_name):
                    actual = getattr(self._tools, func_name)(**args)
                elif env_type == "user" and self._user_tools and hasattr(self._user_tools, func_name):
                    actual = getattr(self._user_tools, func_name)(**args)
                else:
                    continue

                passed = (actual == expected_value)

                # Weight by assertion category
                weight = self._get_assertion_weight(func_name)

                if passed:
                    assertions_passed += 1
                    reward += weight
                else:
                    assertions_failed += 1
                    reward -= weight * 0.5  # failing is penalized at half weight

                assertion_details.append({
                    "name": func_name,
                    "passed": passed,
                    "weight": weight,
                })
            except Exception:
                assertion_details.append({
                    "name": func_name,
                    "passed": False,
                    "weight": 0,
                    "error": True,
                })

        # Store metadata for debugging
        self._reward_breakdown = {
            "tool_matches": tool_matches,
            "tool_total": tool_total,
            "assertions_passed": assertions_passed,
            "assertions_failed": assertions_failed,
            "assertions_total": assertions_total,
            "assertion_details": assertion_details,
            "accumulated_step_reward": self._accumulated_reward,
            "final_reward": reward,
            "total": self._accumulated_reward + reward,
        }

        return reward

    @staticmethod
    def _get_assertion_weight(func_name: str) -> float:
        """
        Get reward weight for an assertion based on its category.

        Mirrors restaurant authority structure:
        - Safety assertions: highest weight (mistakes can be fatal)
        - Authority assertions: high weight (know your limits)
        - Procedural assertions: medium weight (follow SOP)
        - Service quality assertions: lower weight (nice to have)
        """
        # SAFETY — weight 0.5
        safety = [
            "assert_no_unsafe_allergy_recommendation",
            "assert_no_unsafe_allergy_confirmation",
            "assert_allergy_check_performed",
            "assert_allergy_warning_given",
            "assert_plain_water_recommended",
        ]
        if func_name in safety:
            return 0.5

        # AUTHORITY — weight 0.4
        authority = [
            "assert_no_authority_violation",
            "assert_discount_within_server_authority",
            "assert_discount_within_authority",
            "assert_escalated_to_manager",
            "assert_escalated_to_host",
            "assert_no_escalation_made",
            "assert_escalation_made",
            "assert_correct_case_handling",
            "assert_no_internal_issues_exposed",
        ]
        if func_name in authority:
            return 0.4

        # PROCEDURAL — weight 0.25
        procedural = [
            "assert_incident_recorded",
            "assert_no_incident_recorded",
            "assert_reservation_exists",
            "assert_reservation_created",
            "assert_reservation_details_confirmed",
            "assert_availability_checked",
            "assert_kitchen_status_checked",
            "assert_customer_lookup_performed",
            "assert_inventory_checked",
            "assert_secret_code_limit",
            "assert_lunch_special_correctly_applied",
            "assert_discount_applied",
            "assert_party_size_within_capacity",
            "assert_reservation_party_limit",
            "assert_party_size_within_limit",
            "assert_membership_checked_before_offer",
            "assert_appropriate_membership_behavior",
            "assert_escalation_reason_quality",
            "assert_recommended_discount_at_least",
            "assert_recommended_discount_exactly",
            "assert_recommended_action_includes",
            # Policy lookup assertions (agent consulted authoritative data source)
            "assert_policy_looked_up",
            "assert_staff_authority_policy_looked_up",
            "assert_incident_severity_policy_looked_up",
            "assert_allergy_policy_looked_up",
            "assert_service_delay_policy_looked_up",
            "assert_reservation_policy_looked_up",
            "assert_promotion_stacking_policy_looked_up",
            "assert_membership_policy_looked_up",
        ]
        if func_name in procedural:
            return 0.25

        # SERVICE QUALITY — weight 0.15
        # Everything else: compensation, alternatives, communication, etc.
        return 0.15

    def _init_step_tracking(self) -> None:
        """Initialize per-step tracking variables for dense reward."""
        self._prev_unsafe_rec = False
        self._prev_allergy_count = 0
        self._prev_plain_water = False
        self._prev_internal_exposed = False
        self._prev_escalation = False
        self._prev_incident_count = 0
        self._prev_kitchen_checked = False
        self._prev_availability_checked = False
        self._prev_comp_offered = False
        self._prev_complimentary = False
        self._prev_alternative = False
        self._prev_expedited = False
        self._prev_dish_remade = False
        self._accumulated_reward = 0.0
        self._step_reward_signals = {}
        self._reward_breakdown = {}

    # ============== OpenEnv API ==============

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> HospitalityObservation:
        """
        Reset the environment to start a new episode.

        Args:
            seed: Random seed for task selection
            episode_id: Specific episode ID, or a task_id to load a specific task
            **kwargs: Additional arguments (task_id to select a specific task)

        Returns:
            Initial observation with task description and available tools.
        """
        # Create new episode state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Select task
        task_id = kwargs.get("task_id", None)
        if task_id:
            # Find specific task
            task = next((t for t in self._tasks if t["id"] == task_id), None)
            if task is None:
                # Return error observation
                return HospitalityObservation(
                    system_message=f"Task '{task_id}' not found. "
                    f"Available: {self._task_ids[:5]}...",
                    done=True,
                    reward=0.0,
                )
        elif episode_id and episode_id in self._task_ids:
            # episode_id is a task_id
            task = next(t for t in self._tasks if t["id"] == episode_id)
        else:
            # Random task
            if seed is not None:
                random.seed(seed)
            task = random.choice(self._tasks)

        self._current_task = task
        self._init_episode(task)

        # Build system message for the agent
        ticket = task.get("ticket", "")
        description = task.get("description", {})
        category = description.get("category", "")
        purpose = description.get("purpose", "")

        system_msg = (
            f"You are a server at Berkeley Hot Pot restaurant. "
            f"A customer is contacting you ({category}).\n\n"
            f"TICKET: {ticket}\n\n"
            f"Your goal: {purpose}\n\n"
            f"Available tools: {', '.join(self._get_available_tool_names())}\n\n"
            f"{'=' * 60}\n"
            f"STAFF POLICY MANUAL (you MUST follow these rules):\n"
            f"{'=' * 60}\n\n"
            f"{self._policy}"
        )

        # Get the customer's opening message
        user_scenario = task.get("user_scenario", {})
        instructions = user_scenario.get("instructions", {})
        customer_opening = instructions.get("reason_for_call", "")
        known_info = instructions.get("known_info", "")
        if known_info:
            customer_opening = f"{customer_opening} {known_info}"

        self._conversation.append({"role": "customer", "content": customer_opening})

        return HospitalityObservation(
            system_message=system_msg,
            customer_message=customer_opening,
            task_description=ticket,
            available_tools=self._get_available_tool_names(),
            tool_schemas=self._get_tool_schemas(),
            turn_number=0,
            max_turns=MAX_TURNS,
            done=False,
            reward=0.0,
        )

    def step(self, action: HospitalityAction) -> HospitalityObservation:  # type: ignore[override]
        """
        Execute one step: agent sends message and/or calls tool.

        Dense reward: every step returns an immediate reward based on DB state
        changes. Final step adds task-specific assertion checks on top.

        Args:
            action: HospitalityAction with message and/or tool call.

        Returns:
            HospitalityObservation with customer response, tool results, reward.
        """
        self._state.step_count += 1

        # Check if already done
        if self._done:
            return HospitalityObservation(
                system_message="Episode already ended.",
                done=True,
                reward=0.0,
                turn_number=self._state.step_count,
                max_turns=MAX_TURNS,
            )

        # 1. Execute tool call if provided
        tool_result, tool_error = self._execute_tool(action.tool_name, action.tool_args)

        # 2. Record agent message
        if action.message:
            self._conversation.append({"role": "agent", "content": action.message})

        # 3. Compute dense per-step reward (based on DB state changes)
        step_reward = self._compute_step_reward()

        # Penalty for tool errors (called wrong tool or bad args)
        if tool_error:
            step_reward -= 0.1

        self._accumulated_reward += step_reward

        # 4. Check for episode termination
        # a) Max turns reached
        if self._state.step_count >= MAX_TURNS:
            self._done = True
            final_reward = self._compute_final_reward()
            total_reward = step_reward + final_reward
            return HospitalityObservation(
                customer_message="",
                tool_result=tool_result,
                tool_error=tool_error,
                system_message="Maximum turns reached. Episode ended.",
                available_tools=self._get_available_tool_names(),
                turn_number=self._state.step_count,
                max_turns=MAX_TURNS,
                done=True,
                reward=total_reward,
                metadata=self._reward_breakdown,
            )

        # 5. Simulate customer response (only if agent sent a message)
        customer_response = ""
        if action.message:
            customer_response = self._simulate_customer_response(action.message)
            if customer_response:
                self._conversation.append({"role": "customer", "content": customer_response})

            # Check if customer ended conversation
            if "###STOP###" in customer_response:
                self._done = True
                final_reward = self._compute_final_reward()
                total_reward = step_reward + final_reward
                clean_response = customer_response.replace("###STOP###", "").strip()
                return HospitalityObservation(
                    customer_message=clean_response,
                    tool_result=tool_result,
                    tool_error=tool_error,
                    available_tools=self._get_available_tool_names(),
                    turn_number=self._state.step_count,
                    max_turns=MAX_TURNS,
                    done=True,
                    reward=total_reward,
                    metadata=self._reward_breakdown,
                )

        # 6. Return ongoing observation with dense step reward
        return HospitalityObservation(
            customer_message=customer_response,
            tool_result=tool_result,
            tool_error=tool_error,
            available_tools=self._get_available_tool_names(),
            turn_number=self._state.step_count,
            max_turns=MAX_TURNS,
            done=False,
            reward=step_reward,
            metadata=self._step_reward_signals,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
