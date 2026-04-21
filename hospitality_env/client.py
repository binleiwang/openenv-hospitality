# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hospitality Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import HospitalityAction, HospitalityObservation


class HospitalityEnv(
    EnvClient[HospitalityAction, HospitalityObservation, State]
):
    """
    Client for the Hospitality RL Environment.

    Connects to a running HospitalityEnvironment server via WebSocket.

    Example:
        >>> async with HospitalityEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     print(result.observation.customer_message)
        ...
        ...     result = await client.step(HospitalityAction(
        ...         message="I'm sorry to hear that. Let me look up your order.",
        ...         tool_name="get_order_details",
        ...         tool_args={"order_id": "ORD-60-BTH"},
        ...     ))
        ...     print(result.observation.tool_result)
        ...     print(result.observation.customer_message)
    """

    def _step_payload(self, action: HospitalityAction) -> Dict:
        """Convert HospitalityAction to JSON payload."""
        payload = {"message": action.message}
        if action.tool_name:
            payload["tool_name"] = action.tool_name
            payload["tool_args"] = action.tool_args
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[HospitalityObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = HospitalityObservation(
            customer_message=obs_data.get("customer_message", ""),
            tool_result=obs_data.get("tool_result", ""),
            tool_error=obs_data.get("tool_error", ""),
            system_message=obs_data.get("system_message", ""),
            available_tools=obs_data.get("available_tools", []),
            tool_schemas=obs_data.get("tool_schemas", {}),
            task_description=obs_data.get("task_description", ""),
            turn_number=obs_data.get("turn_number", 0),
            max_turns=obs_data.get("max_turns", 20),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
