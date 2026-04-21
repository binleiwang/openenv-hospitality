"""
Tool audit tests for HospitalityTools.

Covers:
1. All 7 new get_*_policy() tools — return structure + side effect tracking
2. Spot-check 5 critical existing tools for correctness + side effects
3. Policy-lookup assertion methods round-trip with the tools

Run:
    pytest tests/test_tools.py -v
"""
import json
from pathlib import Path

import pytest

from hospitality_env.server.domain.data_model import HospitalityDB
from hospitality_env.server.domain.tools import HospitalityTools

DB_PATH = Path(__file__).parent.parent / "hospitality_env" / "server" / "data" / "db.json"


@pytest.fixture
def tools():
    with open(DB_PATH) as f:
        db = HospitalityDB(**json.load(f))
    return HospitalityTools(db)


# ---------- New policy tools ----------

POLICY_TOOLS = [
    ("get_staff_authority_policy", "staff_authority", {"authority_levels", "must_escalate_when"}),
    ("get_incident_severity_policy", "incident_severity", {"cost_severity", "response_by_severity", "red_lines"}),
    ("get_allergy_policy", "allergy", {"core_principle", "safe_vs_unsafe_categories", "red_lines"}),
    ("get_service_delay_policy", "service_delay", {"wait_time_severity"}),
    ("get_reservation_policy", "reservation", {"required_info_to_gather", "party_size_limits"}),
    ("get_promotion_stacking_policy", "promotion_stacking", {"stacking_truth_table", "golden_rule"}),
    ("get_membership_policy", "membership", {"when_to_offer", "when_not_to_offer"}),
]


@pytest.mark.parametrize("tool_name,policy_key,required_keys", POLICY_TOOLS)
def test_policy_tool_returns_structure(tools, tool_name, policy_key, required_keys):
    """Policy tool returns a dict with required keys."""
    fn = getattr(tools, tool_name)
    result = fn()
    assert isinstance(result, dict)
    assert required_keys.issubset(result.keys()), (
        f"{tool_name} missing keys: {required_keys - result.keys()}"
    )


@pytest.mark.parametrize("tool_name,policy_key,_", POLICY_TOOLS)
def test_policy_tool_records_side_effect(tools, tool_name, policy_key, _):
    """Calling a policy tool adds an entry to db.policy_lookups_made."""
    before = len(tools.db.policy_lookups_made)
    getattr(tools, tool_name)()
    after = len(tools.db.policy_lookups_made)
    assert after == before + 1
    assert tools.db.policy_lookups_made[-1]["policy"] == policy_key


@pytest.mark.parametrize("tool_name,policy_key,_", POLICY_TOOLS)
def test_policy_assertion_round_trip(tools, tool_name, policy_key, _):
    """After calling the tool, assert_policy_looked_up returns True."""
    # Before: false
    assert tools.assert_policy_looked_up(policy_key) is False
    # Call the tool
    getattr(tools, tool_name)()
    # After: true
    assert tools.assert_policy_looked_up(policy_key) is True


def test_specific_assertion_helpers(tools):
    """Each named assertion helper matches its policy tool."""
    tools.get_allergy_policy()
    assert tools.assert_allergy_policy_looked_up() is True
    assert tools.assert_staff_authority_policy_looked_up() is False  # unrelated

    tools.get_staff_authority_policy()
    assert tools.assert_staff_authority_policy_looked_up() is True


def test_staff_authority_numeric_values(tools):
    """Spot-check concrete values match policy.md source of truth."""
    r = tools.get_staff_authority_policy()
    assert r["authority_levels"]["server"]["max_discount_pct"] == 12
    assert r["authority_levels"]["server"]["comp_item_limit_usd"] == 10
    assert r["authority_levels"]["host"]["round_off_usd"] == 30


def test_incident_severity_thresholds(tools):
    """Cost thresholds match policy.md."""
    r = tools.get_incident_severity_policy()
    assert r["cost_severity"]["minor"]["max_usd"] == 20
    assert r["cost_severity"]["severe"]["min_usd"] == 200
    assert "children_involved" in r["context_auto_severe"]
    assert "celebration_item_damaged" in r["context_auto_severe"]


def test_promotion_stacking_truth_table(tools):
    """Truth table has the essential cases from policy.md Path G."""
    r = tools.get_promotion_stacking_policy()
    pairs = {(row["a"], row["b"]): row["can_combine"] for row in r["stacking_truth_table"]}
    assert pairs[("voucher", "lunch_special")] is False
    assert pairs[("secret_code", "lunch_special")] is True
    assert pairs[("points_merchandise", "lunch_special")] is True


# ---------- Spot-check critical existing tools ----------

def test_check_allergy_safety_returns_and_tracks(tools):
    """check_allergy_safety returns structured dict + records side effect."""
    # Use first soup base name
    soup_name = tools.db.soup_bases[0].name
    before = len(tools.db.allergy_checks_made)
    result = tools.check_allergy_safety(soup_name, "peanut")
    assert "is_safe" in result
    assert "recommendation" in result
    assert len(tools.db.allergy_checks_made) == before + 1


def test_check_allergy_safety_plain_water_is_safe(tools):
    """Plain Water must always return is_safe=True — safety-critical invariant."""
    result = tools.check_allergy_safety("Plain Water", "peanut")
    assert result["is_safe"] is True
    result2 = tools.check_allergy_safety("Plain Water", "gluten")
    assert result2["is_safe"] is True


def test_get_restaurant_info_returns_required_keys(tools):
    r = tools.get_restaurant_info()
    assert {"name", "location", "hours"}.issubset(r.keys())


def test_get_menu_details_category_filter(tools):
    """Category filter returns only matching items."""
    r = tools.get_menu_details(category="soup_base")
    assert "soup_bases" in r
    # When filtering to soup_base, menu_items should not be included
    assert "menu_items" not in r or not r["menu_items"]


def test_get_current_staff_authority_has_limits(tools):
    r = tools.get_current_staff_authority()
    # Should expose the limits the agent needs to know
    assert any(
        k in r for k in ("max_discount_pct", "discount_limit", "comp_limit", "max_discount")
    ), f"no authority limit field found in {list(r.keys())}"
