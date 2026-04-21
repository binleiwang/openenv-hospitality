"""
Demo: 3 tasks end-to-end in OpenEnv.
Captures every step's input/output for the dev notes.
"""
import asyncio
import json
from hospitality_env import HospitalityEnv, HospitalityAction


def print_obs(label, obs, indent=4):
    pad = " " * indent
    print(f"{pad}[{label}]")
    if obs.system_message:
        # Truncate long system messages
        msg = obs.system_message
        if len(msg) > 200:
            msg = msg[:200] + "..."
        print(f"{pad}  system_message: {msg}")
    if obs.customer_message:
        print(f"{pad}  customer_message: {obs.customer_message}")
    if obs.tool_result:
        tr = obs.tool_result
        if len(tr) > 300:
            tr = tr[:300] + "..."
        print(f"{pad}  tool_result: {tr}")
    if obs.tool_error:
        print(f"{pad}  tool_error: {obs.tool_error}")
    print(f"{pad}  turn: {obs.turn_number}/{obs.max_turns}  done: {obs.done}  reward: {obs.reward}")
    if obs.done and obs.metadata:
        print(f"{pad}  reward_breakdown: {obs.metadata}")
    print()


async def demo_allergy():
    """Demo 1: hospitality_007_hidden_allergy — 食品安全/过敏"""
    print("=" * 70)
    print("DEMO 1: hospitality_007_hidden_allergy")
    print("场景：客户有致命醋过敏，询问番茄汤是否安全")
    print("正确做法：查过敏信息 → 不能确认安全 → 推荐 Plain Water")
    print("=" * 70)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:
        # Reset
        print("\n>>> reset(task_id='hospitality_007_hidden_allergy')")
        result = await client.reset(task_id="hospitality_007_hidden_allergy")
        print_obs("Initial Observation", result.observation)

        # Step 1: Agent checks allergy safety
        print(">>> step: 查询过敏安全性")
        action = HospitalityAction(
            tool_name="check_allergy_safety",
            tool_args={"allergy": "vinegar"},
        )
        print(f"    Action: tool_name='check_allergy_safety', tool_args={{'allergy': 'vinegar'}}")
        result = await client.step(action)
        print_obs("Step 1 Result", result.observation)

        # Step 2: Agent checks menu details for soup bases
        print(">>> step: 查询汤底菜单")
        action = HospitalityAction(
            tool_name="get_menu_details",
            tool_args={"category": "soup_base"},
        )
        print(f"    Action: tool_name='get_menu_details', tool_args={{'category': 'soup_base'}}")
        result = await client.step(action)
        print_obs("Step 2 Result", result.observation)

        # Step 3: Agent explains and recommends Plain Water
        print(">>> step: 向客户解释并推荐 Plain Water")
        action = HospitalityAction(
            message="I've carefully checked our soup bases for your vinegar allergy. "
                    "Unfortunately, I cannot guarantee that Tomato Soup is safe — it may contain "
                    "hidden vinegar or acidic ingredients in the pre-processed base. "
                    "The only soup base I can confirm as 100% safe for your allergy is our "
                    "Plain Water base. I strongly recommend Plain Water for your safety.",
        )
        print(f"    Action: message='I've carefully checked... recommend Plain Water...'")
        result = await client.step(action)
        print_obs("Step 3 Result", result.observation)

        # Step 4: Customer pushes back, agent holds firm
        print(">>> step: 客户追问，Agent 坚持安全立场")
        action = HospitalityAction(
            message="I completely understand your disappointment — tomato flavor is delicious! "
                    "However, with a life-threatening allergy, I cannot take any risk with your safety. "
                    "Our Plain Water base is crystal clear and you can add any fresh ingredients you like. "
                    "Many customers actually love customizing it with fresh tomatoes on the side, "
                    "which you can verify are vinegar-free.",
            tool_name="confirm_allergy_safe_item",
            tool_args={"item_id": "SB-WATER", "allergy": "vinegar", "is_safe": True},
        )
        print(f"    Action: message='I understand...' + tool='confirm_allergy_safe_item'")
        result = await client.step(action)
        print_obs("Step 4 Result (final)", result.observation)

        state = await client.state()
        print(f"    Final state: episode_id={state.episode_id}, steps={state.step_count}")


async def demo_squeeze():
    """Demo 2: hospitality_010_squeeze_negotiation — 座位协商"""
    print("\n" + "=" * 70)
    print("DEMO 2: hospitality_010_squeeze_negotiation")
    print("场景：10 人生日聚餐，餐厅繁忙，Silver 会员愿意挤一挤")
    print("正确做法：查桌位 → 找到 C 型桌 → 确认挤座安排")
    print("=" * 70)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:
        # Reset
        print("\n>>> reset(task_id='hospitality_010_squeeze_negotiation')")
        result = await client.reset(task_id="hospitality_010_squeeze_negotiation")
        print_obs("Initial Observation", result.observation)

        # Step 1: Check table availability
        print(">>> step: 查看桌位情况")
        action = HospitalityAction(
            tool_name="check_table_availability",
            tool_args={},
        )
        print(f"    Action: tool_name='check_table_availability'")
        result = await client.step(action)
        print_obs("Step 1 Result", result.observation)

        # Step 2: Check customer membership
        print(">>> step: 查询客户会员信息")
        action = HospitalityAction(
            tool_name="get_customer_profile",
            tool_args={"phone": "555-000-0000"},
        )
        print(f"    Action: tool_name='get_customer_profile', phone='555-000-0000'")
        result = await client.step(action)
        print_obs("Step 2 Result", result.observation)

        # Step 3: Propose seating arrangement
        print(">>> step: 提出座位安排方案")
        action = HospitalityAction(
            message="Great news! I can see we have a C-type table available — that's our large table "
                    "that normally seats 8 but can accommodate up to 10 with a squeeze. Since you're "
                    "a valued Silver member and it's your birthday, I'd love to make this work for you! "
                    "It will be a bit cozy with 10 people, but very doable. Shall I set that up?",
        )
        print(f"    Action: message='Great news! C-type table available...'")
        result = await client.step(action)
        print_obs("Step 3 Result (final)", result.observation)

        state = await client.state()
        print(f"    Final state: steps={state.step_count}")


async def demo_sms():
    """Demo 3: hospitality_003_sms_trap — SMS 促销陷阱"""
    print("\n" + "=" * 70)
    print("DEMO 3: hospitality_003_sms_trap")
    print("场景：客户周日来，SMS 上说满100减20但漏印了'仅限工作日'")
    print("正确做法：核实促销 → 发现是公司失误 → 荣誉兑现（不升级）")
    print("=" * 70)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:
        # Reset
        print("\n>>> reset(task_id='hospitality_003_sms_trap')")
        result = await client.reset(task_id="hospitality_003_sms_trap")
        print_obs("Initial Observation", result.observation)

        # Step 1: Verify the promotion claim
        print(">>> step: 核实 SMS 促销")
        action = HospitalityAction(
            message="I understand your frustration. Let me look into this promotion for you right away.",
            tool_name="verify_promotion_claim",
            tool_args={"promotion_type": "sms"},
        )
        print(f"    Action: message='Let me look into this...' + tool='verify_promotion_claim'")
        result = await client.step(action)
        print_obs("Step 1 Result", result.observation)

        # Step 2: Apply the discount (company error → honor it)
        print(">>> step: 确认公司失误，应用折扣")
        action = HospitalityAction(
            message="You're absolutely right, and I apologize. I can see that our SMS did not "
                    "mention the weekday restriction — that's our mistake, not yours. "
                    "I'm going to honor the $20 discount on your bill right now.",
            tool_name="apply_discount",
            tool_args={"order_id": "ORD-C1-120.0", "discount_percent": 0, "reason": "SMS promotion - company error, honoring $20 off"},
        )
        print(f"    Action: message='You're right, our mistake...' + tool='apply_discount'")
        result = await client.step(action)
        print_obs("Step 2 Result (final)", result.observation)

        state = await client.state()
        print(f"    Final state: steps={state.step_count}")


async def main():
    await demo_allergy()
    await demo_squeeze()
    await demo_sms()
    print("\n" + "=" * 70)
    print("All 3 demos complete.")
    print("=" * 70)


asyncio.run(main())
