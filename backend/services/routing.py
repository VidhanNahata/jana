DEPARTMENT_MAP = {
    "pothole": "BBMP Roads",
    "road_damage": "BBMP Roads",
    "water_leak": "BWSSB",
    "drainage": "BWSSB",
    "sewage": "BWSSB",
    "streetlight": "BESCOM",
    "electricity": "BESCOM",
    "garbage": "BBMP SWM",
    "waste": "BBMP SWM",
    "tree_fall": "BBMP Horticulture",
    "traffic_signal": "Bangalore Traffic Police",
    "other": "BBMP General"
}

def get_department(category: str) -> str:
    return DEPARTMENT_MAP.get(category, "BBMP General")
