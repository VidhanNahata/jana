import math
from backend.models.master_ticket import MasterTicket
from backend.services.ai_service import get_text_embedding, compute_similarity

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine formula to calculate distance in meters between two points."""
    R = 6371e3 # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# Same category within this distance is the same problem, however it is described
SAME_SPOT_METERS = 50
# Within this distance, descriptions must also be similar
NEARBY_METERS = 500
TEXT_SIMILARITY_THRESHOLD = 0.5


def find_master_ticket(db, description: str, lat: float, lon: float, category: str):
    """
    Find the open master ticket this complaint duplicates, if any.
    Match if same category AND (distance < 50 m, OR distance < 500 m AND
    description similarity > 0.5). The closest same-spot ticket wins, then
    the most similar nearby one.
    """
    open_masters = db.query(MasterTicket).filter(MasterTicket.status != "resolved", MasterTicket.category == category).all()

    new_embedding = get_text_embedding(description)

    best_match = None
    best_key = None
    for master in open_masters:
        dist = calculate_distance(lat, lon, master.centroid_lat, master.centroid_lon)
        if dist >= NEARBY_METERS:
            continue
        sim = compute_similarity(new_embedding, get_text_embedding(master.canonical_description))
        if dist < SAME_SPOT_METERS or sim > TEXT_SIMILARITY_THRESHOLD:
            key = (dist < SAME_SPOT_METERS, -dist if dist < SAME_SPOT_METERS else sim)
            if best_key is None or key > best_key:
                best_key, best_match = key, master

    return best_match
