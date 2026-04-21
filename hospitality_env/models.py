# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Hospitality RL Environment.

The hospitality environment simulates a full-service hot pot restaurant
(Berkeley Hot Pot). The agent acts as a restaurant server handling
customer interactions via phone or in-person.

Action: The agent sends a text message and/or a tool call.
Observation: The environment returns the customer's response, tool results,
             and contextual information.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class HospitalityAction(Action):
    """
    Action the agent (restaurant server) can take.

    The agent can:
    1. Send a text message to the customer
    2. Call a tool (e.g., check_allergy_safety, apply_discount)
    3. Both: send a message AND call a tool in the same step
    """

    message: str = Field(
        default="",
        description="Text message from the agent to the customer. "
        "Can be empty if only calling a tool.",
    )
    tool_name: str = Field(
        default="",
        description="Name of the tool to call (e.g., 'check_allergy_safety'). "
        "Empty string means no tool call.",
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool call as key-value pairs.",
    )


class HospitalityObservation(Observation):
    """
    Observation returned by the environment after each step.

    Contains everything the agent needs to decide its next action.
    Inherits from Observation: done (bool), reward (float|None), metadata (dict).
    """

    # Customer interaction
    customer_message: str = Field(
        default="",
        description="The customer's response message. "
        "Empty on reset (before customer speaks) or after tool-only actions.",
    )

    # Tool results
    tool_result: str = Field(
        default="",
        description="Result of the tool call, if any. "
        "Contains the tool's return value as a formatted string.",
    )
    tool_error: str = Field(
        default="",
        description="Error message if tool call failed (invalid tool name, "
        "wrong arguments, insufficient authority, etc.).",
    )

    # Context
    system_message: str = Field(
        default="",
        description="System instructions or context for the agent. "
        "Populated on reset with the task scenario and policy summary.",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="List of available tool names the agent can call.",
    )
    tool_schemas: Dict[str, Any] = Field(
        default_factory=dict,
        description="Schema for each tool: {tool_name: {description, parameters}}. "
        "Use this to understand what arguments each tool expects.",
    )
    task_description: str = Field(
        default="",
        description="Brief description of the current task/ticket.",
    )

    # Conversation state
    turn_number: int = Field(
        default=0,
        description="Current conversation turn number.",
    )
    max_turns: int = Field(
        default=20,
        description="Maximum allowed turns before episode ends.",
    )
