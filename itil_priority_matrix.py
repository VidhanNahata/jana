import json
from enum import Enum

# --- ITIL Definitions Matrix ---
# Evaluates standard Information Technology Infrastructure Library Incident rules.
# Maps specific combinations of Severity (Impact to civic safety) and Urgency (Time criticality).

class Priority(str, Enum):
    P1 = "P1" # Critical
    P2 = "P2" # High
    P3 = "P3" # Medium
    P4 = "P4" # Low

# Rows = Urgency (High, Medium, Low)
# Columns = Severity (High, Medium, Low)
ITIL_MATRIX = {
    "High": {
        "High": Priority.P1,
        "Medium": Priority.P2,
        "Low": Priority.P3
    },
    "Medium": {
        "High": Priority.P2,
        "Medium": Priority.P3,
        "Low": Priority.P4
    },
    "Low": {
        "High": Priority.P3,
        "Medium": Priority.P4,
        "Low": Priority.P4
    }
}

RESPONSE_SLA = {
    Priority.P1: "Immediate (Dispatch within 1 hr)",
    Priority.P2: "4 hrs",
    Priority.P3: "24 hrs",
    Priority.P4: "72 hrs"
}

def resolve_department(category: str) -> str:
    """Routes structurally to Indian civic domain authorities."""
    dept_map = {
        "Roads": "Public Works Department (PWD)",
        "Sanitation": "Waste Management Authority",
        "Water": "Water Supply & Sewerage Board",
        "Electricity": "State Electricity Board",
        "Other": "General Municipal Admin"
    }
    return dept_map.get(category, "General Municipal Admin")

def get_numeric_rank(priority_str: str) -> int:
    """Helper converting P-scores implicitly into comparative integers constraints (P1 -> 1)"""
    mapping = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
    return mapping.get(priority_str, 4)

def calculate_priority(urgency: str, severity: str, impact_score: float, rf_priority_class: str, category: str = "Other") -> dict:
    """
    Main structural integration function evaluated per ticket inside the FastAPI backend.
    
    Args:
        urgency (str): Time sensitivity ("High", "Medium", "Low")
        severity (str): Absolute magnitude extracted from Visual Pipeline ("High", "Medium", "Low")
        impact_score (float): Numerical domain score calculating systemic density (1.0 - 100.0)
        rf_priority_class (str): ML Priority Prediction dumped by our 100-Tree Random Forest module
        category (str): Target Civic NLP bucket
        
    Returns:
        dict: Standard JSON payload to return bounds natively to UI frameworks.
    """
    
    # 1. Deterministic Matrix Evaluation (Safe Rule-Based limits)
    # Defaulting variables securely to Low structural impact to prevent escalation on malformed payloads
    itil_priority = ITIL_MATRIX.get(urgency, {}).get(severity, Priority.P4).value
    
    # 2. Risk Evaluator Matrix Integration
    rf_rank = get_numeric_rank(rf_priority_class)
    itil_rank = get_numeric_rank(itil_priority)
    
    override_triggered = False
    
    # Take the inherently safer route: Whichever algorithm (AI vs ITIL) scores the hazard as MORE critical, trust it.
    # E.g., If RF machine learning says P3, but the Hardcoded ITIL rules flag it as P1 -> Upgrade to P1.
    final_rank = min(rf_rank, itil_rank)
    
    if final_rank < rf_rank:
        override_triggered = True
        final_priority = itil_priority
    else:
        final_priority = rf_priority_class
        
        # Explicit Edge Case Constraint: If the dynamic impact feature is scaling past 95 on our backend logic, automatically trigger P1 limit unconditionally.
        if impact_score > 95.0 and final_priority != Priority.P1.value:
            final_priority = Priority.P1.value
            override_triggered = True

    # 3. Compile output mappings
    response_payload = {
        "status": "success",
        "data": {
            "evaluation_metrics": {
                "base_urgency": urgency,
                "base_severity": severity,
                "ml_calculated_impact_score": round(impact_score, 2),
                "random_forest_prediction": rf_priority_class,
                "itil_deterministic_anchor": itil_priority
            },
            "final_routing": {
                "final_priority": final_priority,
                "recommended_response_time": RESPONSE_SLA.get(final_priority, "72 hrs"),
                "assigned_department": resolve_department(category),
                "was_ml_overriden": override_triggered
            }
        }
    }
    
    return response_payload

if __name__ == "__main__":
    print("--- Testing FastAPI Backend Component (Task 2H) ---\n")
    
    # Scenario: Random Forest predicted P3 based on limited training data for a "Roads" pothole in low traffic.
    # However, because it was flagged as High Severity (BLIP-2 Visual) and High Urgency (MasterTicket count spiked),
    # The ITIL Matrix intrinsically catches it and triggers an algorithmic override natively upgrading it to P1!
    
    result = calculate_priority(
        urgency="High",
        severity="High",
        impact_score=68.5,       # Moderate ML structural impact
        rf_priority_class="P3",  # The Random Forest got this dangerous situation computationally wrong
        category="Roads"
    )
    
    print(json.dumps(result, indent=4))
