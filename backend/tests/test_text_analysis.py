"""Stage 2: language detection and multilingual problem matching (no model)."""

import pytest

from backend.ai.schema import Language
from backend.ai.text_analysis import analyze_text

CASES = [
    # English
    ("There is a big pothole on 80 feet road, two wheelers are falling", "ROAD.POTHOLE", Language.en),
    ("Garbage has not been collected for a week near the bus stop", "WASTE.UNCOLLECTED", Language.en),
    ("Streetlight not working on our cross since 10 days, very dark at night", "LIGHT.STREETLIGHT_OUT", Language.en),
    ("Water pipe burst and drinking water is leaking on the road", "WATER.PIPE_LEAK", Language.en),
    ("Open manhole near school, children may fall", "DRAIN.OPEN_MANHOLE", Language.en),
    ("Drain is blocked and overflowing onto the street", "DRAIN.BLOCKED", Language.en),
    ("Sewage water is leaking from the underground line, bad smell", "DRAIN.SEWAGE_LEAK", Language.en),
    ("A big tree has fallen across the road after rain", "TREE.FALLEN", Language.en),
    ("Traffic signal at the junction is not working", "TRANSPORT.SIGNAL_FAULT", Language.en),
    ("Bus shelter roof is broken", "TRANSPORT.BUS_SHELTER", Language.en),
    ("Footpath slabs are broken, difficult to walk", "ROAD.FOOTPATH_DAMAGE", Language.en),
    ("Road is waterlogged after rain, knee deep water", "STORMWATER.WATERLOGGING", Language.en),
    ("Live electric wire hanging low near the transformer", "LIGHT.EXPOSED_WIRING", Language.en),
    ("Park benches are broken and the play equipment is damaged", "PARK.BENCH_DAMAGE", Language.en),
    ("Public toilet is dirty and not maintained", "PARK.PUBLIC_TOILET", Language.en),
    ("Wheelchair ramp at the bus stand is broken", "ACCESS.RAMP_DAMAGE", Language.en),
    ("Old building wall is about to collapse on the footpath", "BUILDING.STRUCTURE_DAMAGE", Language.en),
    # Kannada script
    ("ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಗುಂಡಿ ಇದೆ", "ROAD.POTHOLE", Language.kn),
    ("ಕಸ ತೆಗೆದುಕೊಂಡು ಹೋಗಿಲ್ಲ, ವಾಸನೆ ಬರುತ್ತಿದೆ", "WASTE.UNCOLLECTED", Language.kn),
    ("ಬೀದಿ ದೀಪ ಉರಿಯುತ್ತಿಲ್ಲ", "LIGHT.STREETLIGHT_OUT", Language.kn),
    ("ಚರಂಡಿ ಕಟ್ಟಿಕೊಂಡಿದೆ", "DRAIN.BLOCKED", Language.kn),
    ("ನೀರಿನ ಪೈಪ್ ಒಡೆದಿದೆ", "WATER.PIPE_LEAK", Language.kn),
    # Transliterated Kannada and code-mixed
    ("rasthe alli dodda gundi ide", "ROAD.POTHOLE", Language.kn_transliterated),
    ("kasa tegedukondu hogilla", "WASTE.UNCOLLECTED", Language.kn_transliterated),
    ("beedi deepa uriyuttilla", "LIGHT.STREETLIGHT_OUT", Language.kn_transliterated),
    ("namma area alli drainage block agide", "DRAIN.BLOCKED", Language.kn_transliterated),
    ("ನಮ್ಮ road ಅಲ್ಲಿ pothole ಇದೆ", "ROAD.POTHOLE", Language.kn_en_mixed),
]


@pytest.mark.parametrize("text,expected,lang", CASES)
def test_top_candidate_and_language(text, expected, lang):
    t = analyze_text(text)
    assert t.candidates, f"no candidate for {text!r}"
    assert t.candidates[0].subcategory_id == expected
    assert t.language == lang


@pytest.mark.parametrize("text,kind", [
    ("My neighbour has parked his car in front of my gate every day", "PERSONAL_DISPUTE"),
    ("I have not received my pension for three months", "NON_CIVIC_GRIEVANCE"),
    ("Please issue my ration card", "NON_CIVIC_GRIEVANCE"),
    ("Police are not registering my FIR", "NON_CIVIC_GRIEVANCE"),
])
def test_out_of_scope_signals(text, kind):
    t = analyze_text(text)
    assert not t.candidates
    assert t.out_of_scope_signals and t.out_of_scope_signals[0].startswith(kind)


def test_landmark_is_not_the_problem():
    # "bus stop" is where the pothole is, not a bus-shelter complaint
    t = analyze_text("Pothole near the bus stop on MG Road")
    assert t.candidates[0].subcategory_id == "ROAD.POTHOLE"
    assert all(c.subcategory_id != "TRANSPORT.BUS_SHELTER" for c in t.candidates)


def test_empty_text():
    assert analyze_text("   ").status.value == "no_input"


def test_locations_pin_and_duration():
    t = analyze_text("Garbage pile near Jayanagar 4th block, 560041, for the last 2 weeks")
    assert t.location.pin_code == "560041"
    assert t.duration
