"""
End-to-end test of the Hospitality RL Environment.

Tests the full interaction loop:
1. Reset with a specific task
2. Agent calls tools
3. Agent sends messages
4. Customer responds
5. Episode ends with reward
"""
import asyncio
from hospitality_env import HospitalityEnv, HospitalityAction, HospitalityObservation


async def main():
    print("=" * 60)
    print("Hospitality Environment E2E Test")
    print("=" * 60)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:

        # --- Test 1: Reset with specific task ---
        print("\n--- Test 1: Reset (birthday_empathy task) ---")
        result = await client.reset(task_id="hospitality_001_birthday_empathy")
        obs = result.observation
        print(f"  Task: {obs.task_description}")
        print(f"  Customer: {obs.customer_message[:100]}...")
        print(f"  Tools available: {len(obs.available_tools)} tools")
        print(f"  Turn: {obs.turn_number}/{obs.max_turns}")
        print(f"  Done: {obs.done}")
        assert not obs.done
        assert len(obs.available_tools) > 20

        # --- Test 2: Agent calls a tool (no message) ---
        print("\n--- Test 2: Tool call (get_order_details) ---")
        result = await client.step(HospitalityAction(
            tool_name="get_order_details",
            tool_args={"order_id": "ORD-60-BTH"},
        ))
        obs = result.observation
        print(f"  Tool result: {obs.tool_result[:150]}...")
        print(f"  Tool error: {obs.tool_error or 'None'}")
        print(f"  Customer msg: {obs.customer_message or '(none - tool only)'}")
        print(f"  Turn: {obs.turn_number}/{obs.max_turns}")
        assert obs.tool_result  # Should have order details

        # --- Test 3: Agent sends message ---
        print("\n--- Test 3: Agent message ---")
        result = await client.step(HospitalityAction(
            message="I'm so sorry about your father's birthday experience. "
                    "That must have been very disappointing. Let me escalate "
                    "this to our manager right away.",
        ))
        obs = result.observation
        print(f"  Customer response: {obs.customer_message}")
        print(f"  Turn: {obs.turn_number}/{obs.max_turns}")

        # --- Test 4: Agent calls escalation tool with message ---
        print("\n--- Test 4: Tool + message (escalate) ---")
        result = await client.step(HospitalityAction(
            message="I'm recommending a 50% discount on your meal and a personal "
                    "follow-up from our manager. Would that be acceptable?",
            tool_name="escalate_with_solution",
            tool_args={
                "recommended_discount_percent": 50,
                "recommended_actions": ["personal_follow_up", "complimentary_dessert"],
                "reason": "60th birthday dinner ruined - slow service, no greeting",
            },
        ))
        obs = result.observation
        print(f"  Tool result: {obs.tool_result[:100]}...")
        print(f"  Customer: {obs.customer_message}")
        print(f"  Done: {obs.done}")
        print(f"  Reward: {obs.reward}")

        # --- Test 5: Check state ---
        print("\n--- Test 5: Check state ---")
        state = await client.state()
        print(f"  Episode ID: {state.episode_id}")
        print(f"  Steps taken: {state.step_count}")

        # --- Test 6: Invalid tool (after reset to new episode) ---
        print("\n--- Test 6: Reset + invalid tool call ---")
        result = await client.reset(seed=99)
        result = await client.step(HospitalityAction(
            tool_name="nonexistent_tool",
            tool_args={},
        ))
        obs = result.observation
        print(f"  Tool error: {obs.tool_error}")
        assert obs.tool_error  # Should have error

        # --- Test 7: Reset with random task ---
        print("\n--- Test 7: Reset (random task) ---")
        result = await client.reset(seed=42)
        obs = result.observation
        print(f"  New task: {obs.task_description}")
        print(f"  Customer: {obs.customer_message[:100]}...")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


asyncio.run(main())
