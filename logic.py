"""Domain business rules and answer grading. Deterministic, zero tokens.

Kept separate from the tools (utils) and the orchestration (main), so the same
rules drive the workflow, the ground truth, and the grader.
"""

from data import ALTERNATES, INVENTORY


def assess_part(part_id: str, eta_days: int) -> dict:
    """Coverage vs. ETA, shortfall, and the recommended alternate (if at risk)."""
    part = INVENTORY.get(part_id)
    if not part:
        return {"part_id": part_id, "known": False}
    days_of_cover = (part["on_hand"] - part["safety_stock"]) / part["daily_usage"]
    at_risk = days_of_cover < eta_days
    shortfall = max(0, round((eta_days - days_of_cover) * part["daily_usage"]))
    return {
        "part_id": part_id,
        "eta_days": eta_days,
        "days_of_cover": round(days_of_cover, 1),
        "stockout_risk": at_risk,
        "shortfall_units": shortfall,
        "recommended": recommend_supplier(part_id, eta_days, shortfall) if at_risk else None,
    }


def recommend_supplier(part_id: str, eta_days: int, shortfall: int):
    """Among alternates that arrive before the ETA, pick the CHEAPEST.

    Business priority: if an option fits in time, cost wins. Tie-break on speed.
    Note: 'fits in time' means it beats the delayed ETA; it may still dip briefly
    into safety stock. For a stricter rule (no safety-stock dip), filter on
    lead_time_days <= days_of_cover instead.
    """
    candidates = [a for a in ALTERNATES.get(part_id, []) if a["lead_time_days"] < eta_days]
    if not candidates:
        return None
    best = min(candidates, key=lambda a: (a["unit_cost_eur"], a["lead_time_days"]))
    return {
        "supplier": best["supplier"],
        "lead_time_days": best["lead_time_days"],
        "quantity": shortfall,
        "cost_eur": shortfall * best["unit_cost_eur"],
    }


def ground_truth(expected: dict) -> dict:
    """Canonical answer for a scenario from its true delays (part -> eta_days)."""
    at_risk, safe = {}, []
    for part, eta in expected.items():
        facts = assess_part(part, int(eta))
        if facts.get("stockout_risk"):
            at_risk[part] = facts
        else:
            safe.append(part)
    return {"at_risk": at_risk, "safe": safe}


def grade(answer: str, expected: dict) -> dict:
    """Heuristic correctness check against ground truth.

    A scenario PASSES when, for every at-risk part, the answer mentions the part,
    names the cost-optimal supplier that beats the ETA, and states the correct
    order quantity. Supplier choice is now graded against the canonical cheapest
    in-time option, matching the cost-first business priority.
    """
    text = answer.lower()
    gt = ground_truth(expected)
    parts_ok = {}
    for part, facts in gt["at_risk"].items():
        rec = facts.get("recommended") or {}
        supplier = (rec.get("supplier") or "").lower()
        mentioned = part.lower() in text
        supplier_ok = bool(supplier) and supplier in text
        quantity_ok = str(facts["shortfall_units"]) in text
        parts_ok[part] = bool(mentioned and supplier_ok and quantity_ok)
    correct = all(parts_ok.values()) if parts_ok else True
    return {"correct": correct, "parts": parts_ok}
