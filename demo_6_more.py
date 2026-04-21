"""
Demo: 6 more tasks end-to-end (Demo 4-9).
"""
import asyncio
from hospitality_env import HospitalityEnv, HospitalityAction


def pobs(label, obs, indent=4):
    pad = " " * indent
    print(f"{pad}[{label}]")
    if obs.system_message:
        msg = obs.system_message
        if len(msg) > 250:
            msg = msg[:250] + "..."
        print(f"{pad}  system_message: {msg}")
    if obs.customer_message:
        print(f"{pad}  customer_message: {obs.customer_message}")
    if obs.tool_result:
        tr = str(obs.tool_result)
        if len(tr) > 400:
            tr = tr[:400] + "..."
        print(f"{pad}  tool_result: {tr}")
    if obs.tool_error:
        print(f"{pad}  tool_error: {obs.tool_error}")
    print(f"{pad}  turn: {obs.turn_number}/{obs.max_turns}  done: {obs.done}  reward: {obs.reward}")
    if obs.done and obs.metadata:
        print(f"{pad}  reward_breakdown: {obs.metadata}")
    print()


async def demo4():
    """Demo 4: hospitality_009_melted_cake — 冰淇淋蛋糕融化"""
    print("=" * 70)
    print("DEMO 4: hospitality_009_melted_cake")
    print("场景：员工失误导致客户的定制冰淇淋生日蛋糕融化（$100）")
    print("正确做法：查订单 → 记录事故 → 升级到经理（推荐≥50%折扣）")
    print("=" * 70)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:
        print("\n>>> reset(task_id='hospitality_009_melted_cake')")
        r = await client.reset(task_id="hospitality_009_melted_cake")
        pobs("Initial", r.observation)

        # Step 1: Look up order
        print(">>> step 1: 查订单")
        r = await client.step(HospitalityAction(
            message="I'm so sorry about your cake. Let me pull up your order immediately.",
            tool_name="get_order_details",
            tool_args={"order_id": "ORD-CAKE-MELT"},
        ))
        pobs("Step 1", r.observation)

        # Step 2: Record incident
        print(">>> step 2: 记录事故")
        r = await client.step(HospitalityAction(
            tool_name="record_service_incident",
            tool_args={
                "incident_type": "cake_damage",
                "description": "Customer's $100 custom ice cream birthday cake melted due to staff error - not stored in freezer",
                "order_id": "ORD-CAKE-MELT",
                "table_id": "A01",
            },
        ))
        pobs("Step 2", r.observation)

        # Step 3: Escalate to manager
        print(">>> step 3: 升级到经理")
        r = await client.step(HospitalityAction(
            message="This is absolutely our fault and I take full responsibility. "
                    "I'm escalating this to our manager right now. I'm recommending "
                    "a full comp of your cake cost plus a 50% discount on your meal.",
            tool_name="escalate_with_solution",
            tool_args={
                "escalate_to": "manager",
                "reason": "Staff error melted customer's $100 custom birthday cake",
                "recommended_discount_percent": 50,
                "recommended_actions": ["full_cake_reimbursement", "complimentary_dessert"],
            },
        ))
        pobs("Step 3 (final)", r.observation)


async def demo5():
    """Demo 5: hospitality_015_food_poisoning_claim — 食物中毒投诉"""
    print("\n" + "=" * 70)
    print("DEMO 5: hospitality_015_food_poisoning_claim")
    print("场景：客户声称昨天吃完后全家食物中毒，要求全额退款")
    print("正确做法：记录事故 → 表达关心 → 必须升级到经理（不自行处理）")
    print("=" * 70)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:
        print("\n>>> reset(task_id='hospitality_015_food_poisoning_claim')")
        r = await client.reset(task_id="hospitality_015_food_poisoning_claim")
        pobs("Initial", r.observation)

        # Step 1: Express concern + record incident
        print(">>> step 1: 表达关心 + 记录事故")
        r = await client.step(HospitalityAction(
            message="I'm very sorry to hear about your family's illness. Your health is our "
                    "top priority. I'm going to record this immediately and get our manager involved.",
            tool_name="record_service_incident",
            tool_args={
                "incident_type": "food_safety",
                "description": "Customer reports family food poisoning after yesterday's visit - diarrhea and vomiting",
            },
        ))
        pobs("Step 1", r.observation)

        # Step 2: Escalate to manager
        print(">>> step 2: 升级到经理")
        r = await client.step(HospitalityAction(
            message="I've recorded everything. I'm bringing our manager in right now — "
                    "this is too serious for me to handle alone, and you deserve direct "
                    "attention from management.",
            tool_name="escalate_with_solution",
            tool_args={
                "escalate_to": "manager",
                "reason": "Food poisoning claim - family sick after yesterday's visit, requesting full refund",
                "recommended_discount_percent": 100,
                "recommended_actions": ["full_refund", "medical_expense_offer", "follow_up_call"],
            },
        ))
        pobs("Step 2 (final)", r.observation)


async def demo6():
    """Demo 6: hospitality_022_walk_in_vs_reservation — Walk-in 无位"""
    print("\n" + "=" * 70)
    print("DEMO 6: hospitality_022_walk_in_vs_reservation")
    print("场景：周六晚7点Walk-in 4人，餐厅满座，客户看到空桌（其实是预留的）")
    print("正确做法：查桌位 → 解释空桌是预留 → 建议排队等位")
    print("=" * 70)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:
        print("\n>>> reset(task_id='hospitality_022_walk_in_vs_reservation')")
        r = await client.reset(task_id="hospitality_022_walk_in_vs_reservation")
        pobs("Initial", r.observation)

        # Step 1: Check availability
        print(">>> step 1: 查桌位")
        r = await client.step(HospitalityAction(
            message="I completely understand your frustration after driving 30 minutes. "
                    "Let me check our availability for you right away.",
            tool_name="check_table_availability",
            tool_args={"party_size": 4, "date_str": "2026-01-17", "time_str": "19:00"},
        ))
        pobs("Step 1", r.observation)

        # Step 2: Suggest waitlist
        print(">>> step 2: 建议排队")
        r = await client.step(HospitalityAction(
            message="I'm sorry, all our tables are currently occupied or reserved. "
                    "Those tables you see are held for guests who made reservations. "
                    "I'd love to put you on our waitlist — typically the wait is about "
                    "20-30 minutes on a Saturday evening.",
            tool_name="suggest_waitlist",
            tool_args={"reason": "fully_booked"},
        ))
        pobs("Step 2", r.observation)

        # Step 3: Offer something while waiting
        print(">>> step 3: 安抚客户")
        r = await client.step(HospitalityAction(
            message="While you wait, can I offer you complimentary drinks at our bar area? "
                    "We also have a cozy waiting lounge. I'll personally make sure you "
                    "get the next available table.",
        ))
        pobs("Step 3 (final)", r.observation)


async def demo7():
    """Demo 7: hospitality_017_gluten_allergy — 乳糜泻"""
    print("\n" + "=" * 70)
    print("DEMO 7: hospitality_017_gluten_allergy")
    print("场景：乳糜泻客户问哪些汤底无麸质。非会员，8人桌。")
    print("正确做法：查菜单 → 查过敏 → 推荐安全选项 → 推荐入会")
    print("=" * 70)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:
        print("\n>>> reset(task_id='hospitality_017_gluten_allergy')")
        r = await client.reset(task_id="hospitality_017_gluten_allergy")
        pobs("Initial", r.observation)

        # Step 1: Check menu for soup bases
        print(">>> step 1: 查汤底菜单")
        r = await client.step(HospitalityAction(
            message="Absolutely, let me check all our soup bases for gluten content right away.",
            tool_name="get_menu_details",
            tool_args={"category": "soup_base"},
        ))
        pobs("Step 1", r.observation)

        # Step 2: Check allergy safety for Plain Water
        print(">>> step 2: 确认 Plain Water 的无麸质安全性")
        r = await client.step(HospitalityAction(
            tool_name="check_allergy_safety",
            tool_args={"item_id": "SB-WATER", "allergy": "gluten"},
        ))
        pobs("Step 2", r.observation)

        # Step 3: Explain safe options + offer membership
        print(">>> step 3: 说明安全选项 + 推荐入会")
        r = await client.step(HospitalityAction(
            message="For celiac disease, the only soup base I can guarantee is 100% gluten-free "
                    "is our Plain Water base. Other bases may contain hidden wheat or soy-based "
                    "ingredients. With Plain Water, you have full control over what goes in. "
                    "By the way, since you're dining with 8 people, have you heard about "
                    "our membership program? You'd earn points on tonight's meal!",
            tool_name="offer_membership_signup",
            tool_args={"pitch_type": "standard", "benefits_mentioned": ["points_earning", "birthday_perks"]},
        ))
        pobs("Step 3", r.observation)

        # Step 4: Confirm safe item
        print(">>> step 4: 确认安全食品记录")
        r = await client.step(HospitalityAction(
            message="I'll mark Plain Water as confirmed safe for your gluten allergy in our system.",
            tool_name="confirm_allergy_safe_item",
            tool_args={"item_id": "SB-WATER", "allergy": "gluten", "is_safe": True},
        ))
        pobs("Step 4 (final)", r.observation)


async def demo8():
    """Demo 8: hospitality_039_second_code_attempt — 秘密口令"""
    print("\n" + "=" * 70)
    print("DEMO 8: hospitality_039_second_code_attempt")
    print("场景：客户尝试在同一张桌使用多个秘密口令")
    print("正确做法：兑换第一个 → 拒绝第二个（每桌限一个）→ 不升级")
    print("=" * 70)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:
        print("\n>>> reset(task_id='hospitality_039_second_code_attempt')")
        r = await client.reset(task_id="hospitality_039_second_code_attempt")
        pobs("Initial", r.observation)

        # Step 1: Redeem first code
        print(">>> step 1: 兑换第一个口令 'I love your snacks'")
        r = await client.step(HospitalityAction(
            message="Oh, you know the secret phrase! Let me process that for you.",
            tool_name="redeem_secret_code",
            tool_args={"code_phrase": "I love your snacks"},
        ))
        pobs("Step 1", r.observation)

        # Step 2: Customer tries second code
        print(">>> step 2: 兑换第二个口令 'I like your soup base'")
        r = await client.step(HospitalityAction(
            message="Let me check that one as well!",
            tool_name="redeem_secret_code",
            tool_args={"code_phrase": "I like your soup base"},
        ))
        pobs("Step 2", r.observation)

        # Step 3: Explain policy
        print(">>> step 3: 解释规则")
        r = await client.step(HospitalityAction(
            message="I'm sorry, but our secret code program allows one code per table per visit. "
                    "You've already redeemed one today! But we'd love to see you again — "
                    "you can use a different code on your next visit.",
        ))
        pobs("Step 3 (final)", r.observation)


async def demo9():
    """Demo 9: hospitality_029_mlk_day_lunch_special — MLK 日午餐特价"""
    print("\n" + "=" * 70)
    print("DEMO 9: hospitality_029_mlk_day_lunch_special")
    print("场景：客户在 MLK Day 要午餐特价，周一下午5点前，但联邦假日不适用")
    print("正确做法：查午餐特价可用性 → 解释假日不适用 → 不升级")
    print("=" * 70)

    async with HospitalityEnv(base_url="http://127.0.0.1:8000") as client:
        print("\n>>> reset(task_id='hospitality_029_mlk_day_lunch_special')")
        r = await client.reset(task_id="hospitality_029_mlk_day_lunch_special")
        pobs("Initial", r.observation)

        # Step 1: Check lunch special availability
        print(">>> step 1: 查午餐特价可用性")
        r = await client.step(HospitalityAction(
            message="Let me check if the lunch special is available today.",
            tool_name="check_lunch_special_availability",
            tool_args={},
        ))
        pobs("Step 1", r.observation)

        # Step 2: Explain holiday exclusion
        print(">>> step 2: 解释假日政策")
        r = await client.step(HospitalityAction(
            message="I understand your frustration — it is Monday and it is before 5pm, "
                    "so it seems like it should apply. However, today is Martin Luther King Jr. Day, "
                    "which is a federal holiday. Our lunch special is not available on federal holidays. "
                    "I'm sorry for the inconvenience. Can I help you with our regular menu instead?",
        ))
        pobs("Step 2", r.observation)

        # Step 3: Continue helping
        print(">>> step 3: 继续服务")
        r = await client.step(HospitalityAction(
            message="Our regular menu has some great value options too! Would you like me "
                    "to recommend some popular dishes in a similar price range?",
        ))
        pobs("Step 3 (final)", r.observation)


async def main():
    await demo4()
    await demo5()
    await demo6()
    await demo7()
    await demo8()
    await demo9()
    print("\n" + "=" * 70)
    print("All 6 demos (4-9) complete.")
    print("=" * 70)


asyncio.run(main())
