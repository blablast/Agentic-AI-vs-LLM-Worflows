"""Domain data and configuration for the workflow-vs.-agent benchmark.

Pure data, no logic. Edit scenarios and inventory here.
"""

TEMPERATURE = 0.0  # clean signal across the complexity axis

INVENTORY = {
    "CTRL-9000": {"on_hand": 420, "daily_usage": 60, "safety_stock": 120},
    "HMI-22":    {"on_hand": 900, "daily_usage": 60, "safety_stock": 150},
    "PSU-5":     {"on_hand": 1300, "daily_usage": 60, "safety_stock": 200},
    "MOT-3":     {"on_hand": 300, "daily_usage": 50, "safety_stock": 100},
    "SNS-7":     {"on_hand": 240, "daily_usage": 40, "safety_stock": 80},
}

ALTERNATES = {
    "CTRL-9000": [{"supplier": "AltParts GmbH", "lead_time_days": 6, "unit_cost_eur": 110}],
    "MOT-3":     [{"supplier": "MotorWorks", "lead_time_days": 5, "unit_cost_eur": 90},
                  {"supplier": "FastMotors", "lead_time_days": 3, "unit_cost_eur": 130}],
    "SNS-7":     [{"supplier": "SensorHub", "lead_time_days": 7, "unit_cost_eur": 40}],
}

# Data for the two tangential tools. Indirectly useful, not required for the
# core risk/quantity/cost deliverable. A focused agent should mostly skip them.
SUPPLIER_RATINGS = {
    "AltParts GmbH": 92, "MotorWorks": 81, "FastMotors": 87,
    "SensorHub": 90, "PanelTech": 84,
}

PART_DETAILS = {
    "CTRL-9000": {"category": "controller", "criticality": "high"},
    "HMI-22":    {"category": "panel", "criticality": "medium"},
    "PSU-5":     {"category": "power supply", "criticality": "low"},
    "MOT-3":     {"category": "stepper motor", "criticality": "high"},
    "SNS-7":     {"category": "sensor", "criticality": "medium"},
}

# Each scenario carries the ground-truth delays (part -> eta_days) so the grader
# does not depend on the model's own extraction.
SCENARIOS = [
    {"id": "S1-simple", "expected": {"CTRL-9000": 9}, "email": """
Hi team, sorry for the late update. Due to a sub-supplier failure, the CTRL-9000
controller from PO-2024-0317 will be delayed; new ETA is 9 days from today. The
other items are on schedule. Best, Marek."""},

    {"id": "S2-simple", "expected": {"MOT-3": 7}, "email": """
Hello, quick heads-up: the MOT-3 stepper motor shipment slips by 7 days from
today due to a customs hold. Everything else is unaffected. Regards, Supplier."""},

    {"id": "S3-medium", "expected": {"CTRL-9000": 9, "PSU-5": 5}, "email": """
Hi, two updates on PO-2024-0402. The CTRL-9000 controller is delayed to 9 days
from today, and the PSU-5 power supply to 5 days from today. HMI-22 panels ship
on time. Let us know if this is a problem. Thanks, Marek."""},

    {"id": "S4-hard", "expected": {"MOT-3": 8, "SNS-7": 11}, "email": """
Hi team, apologies, this one is messy. The MOT-3 motors are stuck in transit and
realistically land early next week, call it about 8 days out. The SNS-7 sensors
are also slipping, our planner says roughly a week and a half, so plan for 11
days. If either is critical, let us know and we will see what we can do.
Cheers, Supplier."""},

    {"id": "S5-harder", "expected": {"CTRL-9000": 9, "MOT-3": 8, "HMI-22": 6}, "email": """
Hello, several items from PO-2024-0511 are affected. CTRL-9000 controllers move
to 9 days from today, MOT-3 motors to 8 days, and HMI-22 panels to 6 days. We
understand some of these are tight against your line. If you need faster options
on any of them, you are welcome to source from alternates. Regards, Marek."""},
]

TASK = "Handle this supplier delay email."

# Shared output contract: BOTH approaches must produce the SAME deliverable, in
# the SAME strict format, so the comparison is apples-to-apples.
DELIVERABLE = (
    "Required output: ONE line per delayed part, in exactly this format:\n"
    "  <PART-ID>: AT RISK | supplier: <name> | qty: <N> units | cost: EUR <N>\n"
    "For a part that is NOT at risk, use exactly:\n"
    "  <PART-ID>: OK | no action\n"
    "For each AT-RISK part, among the alternate suppliers that arrive before the "
    "ETA choose the CHEAPEST one (if it fits in time, cost is the priority; break "
    "ties by speed). Do not add any extra commentary, headers, or explanation."
)

# Token pricing for real-cost estimation. Output is the expensive side across all
# majors (~5-6x input): Anthropic 1:5 ($3/$15), OpenAI ~1:6 ($5/$30), Google ~1:6
# ($2/$12). Normalizing output to 1.0, the input share averages ~0.18.
OUTPUT_COST = 1.0
INPUT_COST_RATIO = 0.18
