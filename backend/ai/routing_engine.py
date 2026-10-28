"""
Stage 5: Karnataka civic-authority and jurisdiction routing.

Deterministic rules over the versioned registry (data/authority_registry.json)
and gazetteer; the language model is never asked who is responsible. Inputs:
the classified subcategory, place mentions from the text, citizen-provided
fields (locality / city / district / PIN / corporation / asset owner) and the
GPS fix. Unverified registry facts always lead to a confirmation request, and
an infrastructure type with no configured authority returns
authority_not_configured instead of an invented destination.
"""

import math
from dataclasses import dataclass, field

from backend.ai import knowledge
from backend.ai.schema import (AuthorityAlternative, Classification, Jurisdiction, LocationMentions, Routing,
                               RoutingStatus, UrbanRural)

BENGALURU_CORPS = ["KA.BLR.CENTRAL", "KA.BLR.NORTH", "KA.BLR.SOUTH", "KA.BLR.EAST", "KA.BLR.WEST"]
ESCOMS = ["KA.ESCOM.BESCOM", "KA.ESCOM.CESC", "KA.ESCOM.HESCOM", "KA.ESCOM.GESCOM", "KA.ESCOM.MESCOM"]
ELECTRICAL_DISTRIBUTION = {"LIGHT.EXPOSED_WIRING", "LIGHT.TRANSFORMER_ISSUE", "LIGHT.POLE_DAMAGE"}
WATER_SEWER = {"WATER.PIPE_LEAK", "WATER.TAP_DAMAGE", "WATER.SUPPLY_INFRA_DAMAGE", "DRAIN.SEWAGE_LEAK",
               "DRAIN.OPEN_MANHOLE"}


@dataclass
class CitizenLocation:
    """Location details the citizen typed or confirmed in the form (all optional)."""
    locality: str | None = None
    city: str | None = None
    district: str | None = None
    pin_code: str | None = None
    city_corporation_id: str | None = None   # e.g. KA.BLR.SOUTH, chosen by the citizen
    asset_owner_id: str | None = None        # e.g. KA.BMTC when the citizen knows who owns the asset
    rural: bool | None = None
    lat: float | None = None
    lon: float | None = None


@dataclass
class _Decision:
    status: RoutingStatus
    authority_id: str | None = None
    rationale: list[str] = field(default_factory=list)
    alternatives: list[tuple[str, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _km(a, b):
    dlat, dlon = math.radians(b[0] - a[0]), math.radians(b[1] - a[1])
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def resolve_jurisdiction(mentions: LocationMentions, citizen: CitizenLocation) -> Jurisdiction:
    gaz = knowledge.gazetteer()
    j = Jurisdiction()
    pin = citizen.pin_code or mentions.pin_code
    if pin:
        j.pin_code = pin
    # 1. Explicit citizen fields win, then the complaint text, then PIN, then approximate GPS
    city = citizen.city or mentions.city
    district = citizen.district or mentions.district
    if citizen.city or citizen.district:
        j.basis.append("citizen-provided city/district")
    elif mentions.city or mentions.district:
        j.basis.append("place named in the complaint text")
    if not city and pin and pin.startswith(gaz["bengaluru_pin_prefix"]):
        city, district = "Bengaluru", district or "Bengaluru Urban"
        j.basis.append(f"PIN code {pin} (560xxx postal region)")
    if not city and citizen.lat is not None and citizen.lon is not None:
        for c in gaz["corporation_cities"]:
            if _km((citizen.lat, citizen.lon), c["centre"]) <= c["approx_radius_km"]:
                city, district = c["city"], district or c["district"]
                j.basis.append(f"GPS within ~{c['approx_radius_km']} km of {c['city']} centre (approximate)")
                break
    # Canonical city name ("Hubli", "Bangalore" -> registry spelling) and its district
    if city:
        key = city.strip().lower()
        for c in gaz["corporation_cities"]:
            if key == c["city"].lower() or key in c.get("aliases", []):
                city, district = c["city"], district or c["district"]
                break
    j.city, j.district = city, district
    rural =citizen.rural if citizen.rural is not None else mentions.rural
    if city:
        j.urban_rural = UrbanRural.urban
    elif rural:
        j.urban_rural = UrbanRural.rural
        j.basis.append("rural place markers (village / panchayat / taluk)")
    # City/district named by the citizen, the text or a PIN is taken as reliable at city level;
    # a GPS-radius guess is approximate. (Ward/panchayat boundaries are never available offline.)
    j.approximate = not any(b.startswith(("citizen", "place named", "PIN")) for b in j.basis)
    return j


def _local_body(j: Jurisdiction, citizen: CitizenLocation, sub_id: str) -> _Decision:
    if j.city == "Bengaluru":
        if citizen.city_corporation_id in BENGALURU_CORPS:
            return _Decision(RoutingStatus.confirmation_required, citizen.city_corporation_id,
                             ["Citizen selected the Bengaluru city corporation for this locality.",
                              "Corporation boundaries are not verified in the registry; officials should confirm."])
        return _Decision(RoutingStatus.confirmation_required, None,
                         ["Bengaluru: civic maintenance is handled by one of the five GBA city corporations.",
                          "Verified ward-to-corporation boundaries are not available offline, so the citizen must "
                          "choose the corporation for the locality (BBMP is a legacy name, not a destination)."],
                         alternatives=[(c, "Possible city corporation for this locality") for c in BENGALURU_CORPS]
                         + [("KA.BLR.GBA", "Coordination/escalation if the corporation is unknown")],
                         missing=["city_corporation"])
    if j.city:
        corp = next((c for c in knowledge.gazetteer()["corporation_cities"] if c["city"] == j.city), None)
        if corp and corp["authority"] != "BENGALURU_CORPORATIONS":
            return _Decision(RoutingStatus.confirmation_required, corp["authority"],
                             [f"{j.city} is served by its city corporation (registry entry unverified)."],
                             alternatives=[("KA.DMA", "Administrative escalation for urban local bodies")])
    if j.urban_rural == UrbanRural.rural:
        return _Decision(RoutingStatus.authority_not_configured, "KA.GP.GENERIC",
                         ["Rural location: village roads, drains, streetlights and water points are normally the "
                          "gram panchayat's.", "Individual gram panchayats are not in the registry yet."],
                         alternatives=[("KA.RDPR", "Escalation for rural local bodies")],
                         missing=["village", "taluk", "district"])
    if j.district:
        return _Decision(RoutingStatus.authority_not_configured, "KA.ULB.GENERIC",
                         [f"Location is in {j.district} district but the town's municipal council / town panchayat "
                          "is not in the registry yet."],
                         alternatives=[("KA.DMA", "Administrative escalation for urban local bodies")],
                         missing=["town_or_city"])
    return _Decision(RoutingStatus.location_required, None,
                     ["The town, city or locality could not be determined from the complaint, PIN or GPS."],
                     missing=["locality_or_landmark", "town_or_city"])


def _escom_for(district: str | None) -> str | None:
    if not district:
        return None
    for eid in ESCOMS:
        a = knowledge.authority(eid)
        if district in a["jurisdiction"].get("districts", []):
            return eid
    return None


def _decide(sub_id: str, j: Jurisdiction, mentions: LocationMentions, citizen: CitizenLocation) -> _Decision:
    markers = set(mentions.asset_markers)

    # 0. The citizen knows the asset owner (e.g. picked BMTC for a bus shelter)
    if citizen.asset_owner_id:
        if knowledge.handles(citizen.asset_owner_id, sub_id):
            return _Decision(RoutingStatus.confirmation_required, citizen.asset_owner_id,
                             ["Citizen identified the agency that owns or maintains this asset."])
        return _Decision(RoutingStatus.confirmation_required, None,
                         [f"The selected agency ({citizen.asset_owner_id}) is not responsible for "
                          f"{knowledge.subcategory(sub_id)['name_en'].lower()} in the registry; choose another."],
                         missing=["asset_owner"])

    # 1. Asset-specific owners named in the complaint
    if "metro" in markers and knowledge.handles("KA.BMRCL", sub_id):
        return _Decision(RoutingStatus.resolved, "KA.BMRCL", ["The complaint concerns a Namma Metro station/facility."])
    if "ksrtc" in markers and knowledge.handles("KA.KSRTC", sub_id):
        return _Decision(RoutingStatus.resolved, "KA.KSRTC", ["The complaint concerns a KSRTC bus station/facility."])
    if "transmission" in markers and knowledge.handles("KA.KPTCL", sub_id):
        return _Decision(RoutingStatus.confirmation_required, "KA.KPTCL",
                         ["A transmission line/tower or substation is mentioned (KPTCL asset)."],
                         alternatives=[(e, "If it is a local distribution line") for e in filter(None, [_escom_for(j.district)])])
    if "forest" in markers and knowledge.handles("KA.FOREST", sub_id):
        return _Decision(RoutingStatus.confirmation_required, "KA.FOREST",
                         ["The citizen states the tree is on forest land."])
    if mentions.road_class == "national_highway" and knowledge.handles("IN.NHAI", sub_id):
        return _Decision(RoutingStatus.confirmation_required, "IN.NHAI",
                         ["The complaint names a national highway.",
                          "Some NH stretches are maintained by the PWD NH division or a concessionaire; confirm."],
                         alternatives=[("KA.PWD", "If this NH stretch is maintained by the PWD NH division")])
    if mentions.road_class == "state_highway" and knowledge.handles("KA.PWD", sub_id):
        return _Decision(RoutingStatus.confirmation_required, "KA.PWD", ["The complaint names a state highway."])

    local = _local_body(j, citizen, sub_id)

    # 2. Electricity distribution hazards -> the ESCOM serving the district
    if sub_id in ELECTRICAL_DISTRIBUTION:
        escom = _escom_for(j.district)
        if not escom:
            return _Decision(RoutingStatus.location_required, None,
                             ["Electrical hazards go to the electricity supply company serving the district, "
                              "but the district is unknown."], missing=["district"])
        a = knowledge.authority(escom)
        dec = _Decision(RoutingStatus.resolved if a["jurisdiction"]["verification"] == "content_verified"
                        else RoutingStatus.confirmation_required, escom,
                        [f"Electricity distribution infrastructure in {j.district} district is maintained by "
                         f"{a['official_name']}."])
        if a["jurisdiction"]["verification"] != "content_verified":
            dec.rationale.append("The company's district service area is not verified in the registry.")
        if sub_id == "LIGHT.POLE_DAMAGE" and local.authority_id:
            dec.alternatives.append((local.authority_id, "If it is a streetlight pole owned by the local body"))
        return dec

    # 3. Water supply and sewerage
    if sub_id in WATER_SEWER:
        if j.city == "Bengaluru":
            dec = _Decision(RoutingStatus.resolved, "KA.BWSSB",
                            ["In Bengaluru, piped water supply and the sewerage network are BWSSB's."])
            if sub_id == "DRAIN.OPEN_MANHOLE":
                dec.status = RoutingStatus.confirmation_required
                dec.rationale.append("An open manhole may belong to the sewer network (BWSSB) or a stormwater "
                                     "drain (city corporation); confirm which.")
                dec.alternatives += [(c, "If it is a stormwater-drain manhole") for c in BENGALURU_CORPS]
            return dec
        if local.authority_id and local.authority_id not in ("KA.GP.GENERIC",):
            local.rationale.append("Outside Bengaluru, day-to-day water and sewer maintenance is normally the "
                                   "urban local body's; KUWSDB only for schemes it owns.")
            local.alternatives.append(("KA.KUWSDB", "If the asset is part of a KUWSDB scheme"))
        return local

    # 4. Bengaluru specifics where another agency may own the asset
    if j.city == "Bengaluru":
        if sub_id == "TRANSPORT.BUS_SHELTER":
            local.alternatives.insert(0, ("KA.BMTC", "If BMTC owns or maintains this bus shelter"))
            local.rationale.append("Bus shelters may be maintained by the corporation or BMTC; confirm the owner.")
        if sub_id.startswith("WASTE."):
            local.alternatives.append(("KA.BSWML", "Bengaluru Solid Waste Management Ltd (division of responsibility "
                                                   "with the corporations unverified)"))
        if "bda_layout" in markers:
            local.alternatives.insert(0, ("KA.BDA", "If the asset is in a BDA layout not handed over to the corporation"))
    if sub_id == "LIGHT.STREETLIGHT_OUT":
        escom = _escom_for(j.district)
        local.rationale.append("Streetlights are normally maintained by the local body that owns them, not the "
                               "electricity company.")
        if escom:
            local.alternatives.append((escom, "If the streetlight is on an electricity-company pole/feeder"))
    if sub_id.startswith("ROAD.") and local.authority_id:
        local.alternatives.append(("KA.PWD", "If the road is a state highway or major district road maintained by PWD"))
        local.rationale.append("Road ownership is not verified; local roads are assumed to be the local body's.")
    if sub_id.startswith("STORMWATER."):
        local.rationale.append("Stormwater drains and road waterlogging are the local body's, not the water board's.")
    if sub_id.startswith("TREE.") and local.authority_id:
        local.rationale.append("Street trees are maintained by the local body; the Forest Department only on forest land.")
    if sub_id == "TRANSPORT.SIGNAL_FAULT":
        local.rationale.append("Traffic-signal maintenance responsibility is not configured in the registry; "
                               "officials should confirm the agency.")
        local.status = RoutingStatus.confirmation_required
    return local


def route(classification: Classification, mentions: LocationMentions, citizen: CitizenLocation) -> Routing:
    reg_version = knowledge.registry()["version"]
    if classification.out_of_scope or not classification.subcategory_id:
        return Routing(registry_version=reg_version, status=RoutingStatus.not_applicable,
                       routing_rationale=["Out of scope: no civic authority is selected."],
                       jurisdiction_confirmation_required=False)

    j = resolve_jurisdiction(mentions, citizen)
    d = _decide(classification.subcategory_id, j, mentions, citizen)
    a = knowledge.authority(d.authority_id) if d.authority_id else None
    alternatives = []
    for aid, why in d.alternatives:
        alt = knowledge.authority(aid)
        if alt and aid != d.authority_id and aid not in [x.authority_id for x in alternatives]:
            alternatives.append(AuthorityAlternative(authority_id=aid, authority_name=alt["official_name"], reason=why))

    url_status = a["verification"]["url"] if a else None
    authority_verification = (a["jurisdiction"].get("verification") if a else None)
    confidence = {RoutingStatus.resolved: 0.85, RoutingStatus.confirmation_required: 0.5,
                  RoutingStatus.authority_not_configured: 0.3, RoutingStatus.location_required: 0.1}[d.status]
    if d.status == RoutingStatus.resolved and authority_verification != "content_verified":
        confidence = 0.7
    return Routing(
        registry_version=reg_version,
        status=d.status,
        authority_id=d.authority_id,
        authority_name=a["official_name"] if a else None,
        authority_name_kn=a.get("name_kn") if a else None,
        authority_type=a.get("authority_type") if a else None,
        jurisdiction=j,
        verified_url=a.get("official_source_url") if a and url_status == "url_verified" else None,
        url_verification=url_status,
        authority_verification=authority_verification,
        routing_rationale=d.rationale + ([f"Location basis: {', '.join(j.basis)}."] if j.basis else []),
        confidence=confidence,
        alternatives=alternatives,
        jurisdiction_confirmation_required=d.status != RoutingStatus.resolved or j.approximate,
        missing_location_fields=d.missing,
        escalation_authority_id=a.get("escalation") if a else None,
    )
