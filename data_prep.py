"""Convert train.csv's fixed-order C1..C8 questions into test.csv-style
shuffled, plain-numbered questions, with a synthetic grounded chain-of-thought
target ending in \\boxed{N}.

Pure pandas/regex — no GPU or model needed. Run directly (`python data_prep.py`)
to sanity-check parsing against the real CSVs.
"""
import math
import random
import re

import pandas as pd

C_OPTION_RE = re.compile(r"^C(\d+):\s*(.+)$", re.MULTILINE)
# test.csv/validation_questions.csv use a randomized label per question: plain
# digits ("1:"), or a single letter + digit ("M1:"), possibly with leading
# whitespace. Label is any alnum token so this covers all observed styles.
PLAIN_OPTION_RE = re.compile(r"^\s*([A-Za-z0-9]+):\s*(.+)$", re.MULTILINE)
OPTION_COUNT_RE = re.compile(r"following (\d+) potential (root causes|solutions)")

DRIVE_TABLE_MARKER = "User plane drive test data"
ENG_TABLE_MARKER = "Engeneering parameters data"

# Calibrated against test.csv's actual telecom-canonical subset: most questions
# show all 8 options, but a meaningful fraction show a random subset of 5-7.
SUBSET_SIZE_WEIGHTS = {8: 0.786, 6: 0.082, 5: 0.072, 7: 0.060}
# ~48% of test questions label options with plain digits; the rest use a
# single random uppercase letter + digit (e.g. "M1", "M2", ...).
PREFIX_DIGIT_PROB = 0.48
LETTERS = [chr(ord("A") + i) for i in range(26)]

NEIGHBOR_PCI_COLS = [
    f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI" for i in range(1, 6)
]
NEIGHBOR_BRSRP_COLS = [
    f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]"
    for i in range(1, 6)
]

CANONICAL_DESCRIPTIONS = {
    "C1": "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    "C2": "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    "C3": "A neighboring cell provides higher throughput.",
    "C4": "Non-colocated co-frequency neighboring cells cause severe overlapping coverage.",
    "C5": "Frequent handovers degrade performance.",
    "C6": "Neighbor cell and serving cell have the same PCI mod 30, leading to interference.",
    "C7": "Test vehicle speed exceeds 40km/h, impacting user throughput.",
    "C8": "Average scheduled RBs are below 160, affecting throughput.",
}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_options(question_text: str):
    """Return list of (label, description) in on-page order.

    label is 'C1'..'C8' for canonical (train.csv) questions, or the plain
    number string ('1', '2', ...) for shuffled test/validation questions.
    """
    c_matches = C_OPTION_RE.findall(question_text)
    if c_matches:
        return [(f"C{n}", desc.strip()) for n, desc in c_matches]

    header_end = len(question_text)
    for marker in ("\nGiven:", "\n" + DRIVE_TABLE_MARKER, "\n" + ENG_TABLE_MARKER):
        idx = question_text.find(marker)
        if idx != -1:
            header_end = min(header_end, idx)
    header = question_text[:header_end]
    return [(n, desc.strip()) for n, desc in PLAIN_OPTION_RE.findall(header)]


def _extract_table(question_text: str, marker: str):
    idx = question_text.find(marker)
    if idx == -1:
        return None
    lines = question_text[idx:].split("\n")[1:]
    table_lines, started = [], False
    for line in lines:
        if "|" in line:
            table_lines.append(line)
            started = True
        elif started:
            break
    if not table_lines:
        return None
    header = [c.strip() for c in table_lines[0].split("|")]
    rows = [l.split("|") for l in table_lines[1:]]
    df = pd.DataFrame(rows, columns=header).replace("-", pd.NA)
    return df


def parse_drive_test_table(question_text: str):
    return _extract_table(question_text, DRIVE_TABLE_MARKER)


def parse_engineering_params(question_text: str):
    return _extract_table(question_text, ENG_TABLE_MARKER)


# --------------------------------------------------------------------------
# Feature computation (for grounding synthetic CoT, not for prediction)
# --------------------------------------------------------------------------

def _num(series_or_val, default=None):
    v = pd.to_numeric(series_or_val, errors="coerce")
    return v


def _vertical_beamwidth(scenario):
    if not isinstance(scenario, str):
        return None
    s = scenario.strip().upper()
    if s == "DEFAULT" or re.match(r"SCENARIO_[1-5]$", s):
        return 6.0
    if re.match(r"SCENARIO_(6|7|8|9|10|11)$", s):
        return 12.0
    if re.match(r"SCENARIO_\d+$", s):
        return 25.0
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def compute_features(drive_df, eng_df):
    feats = {}
    if drive_df is None or drive_df.empty:
        return feats

    speed = _num(drive_df.get("GPS Speed (km/h)"))
    feats["avg_speed"] = round(speed.mean(), 1) if speed is not None and speed.notna().any() else None
    feats["max_speed"] = round(speed.max(), 1) if speed is not None and speed.notna().any() else None

    rb = _num(drive_df.get("5G KPI PCell Layer1 DL RB Num (Including 0)"))
    feats["avg_rb"] = round(rb.mean(), 1) if rb is not None and rb.notna().any() else None

    serving_pci = _num(drive_df.get("5G KPI PCell RF Serving PCI"))
    if serving_pci is not None and serving_pci.notna().any():
        transitions = int((serving_pci != serving_pci.shift()).sum()) - 1
        feats["handover_count"] = max(transitions, 0)
        mode = serving_pci.mode()
        dom_pci = int(mode.iloc[0]) if not mode.empty else None
    else:
        feats["handover_count"] = None
        dom_pci = None
    feats["serving_pci"] = dom_pci

    serving_rsrp = _num(drive_df.get("5G KPI PCell RF Serving SS-RSRP [dBm]"))
    feats["avg_serving_rsrp"] = (
        round(serving_rsrp.mean(), 1) if serving_rsrp is not None and serving_rsrp.notna().any() else None
    )

    neighbor_pairs = []
    for pcicol, brcol in zip(NEIGHBOR_PCI_COLS, NEIGHBOR_BRSRP_COLS):
        if pcicol in drive_df.columns and brcol in drive_df.columns:
            pcis = _num(drive_df[pcicol])
            brs = _num(drive_df[brcol])
            for pci, br in zip(pcis, brs):
                if pd.notna(pci) and pd.notna(br):
                    neighbor_pairs.append((int(pci), float(br)))

    feats["best_neighbor_brsrp"] = round(max(b for _, b in neighbor_pairs), 1) if neighbor_pairs else None

    collisions = set()
    if dom_pci is not None:
        dom_mod = dom_pci % 30
        collisions = {p for p, _ in neighbor_pairs if p != dom_pci and p % 30 == dom_mod}
    feats["pci_mod30_collisions"] = sorted(collisions)

    comparable = set()
    if feats["avg_serving_rsrp"] is not None:
        comparable = {p for p, b in neighbor_pairs if abs(b - feats["avg_serving_rsrp"]) <= 6}
    feats["comparable_strength_neighbors"] = sorted(comparable)

    feats["downtilt_total"] = None
    feats["vertical_beamwidth"] = None
    feats["height"] = None
    feats["distance_km"] = None
    feats["serving_gnodeb"] = None
    feats["non_colocated_overlap_pcis"] = []

    if eng_df is not None and not eng_df.empty and dom_pci is not None and "PCI" in eng_df.columns:
        eng_pci = _num(eng_df["PCI"])
        match = eng_df[eng_pci == dom_pci]
        if not match.empty:
            row = match.iloc[0]
            mech = _num(pd.Series([row.get("Mechanical Downtilt")])).iloc[0]
            dig = _num(pd.Series([row.get("Digital Tilt")])).iloc[0]
            dig_interp = None
            if pd.notna(dig):
                dig_interp = 6.0 if float(dig) == 255 else float(dig)
            if pd.notna(mech) and dig_interp is not None:
                feats["downtilt_total"] = round(float(mech) + dig_interp, 1)
            feats["vertical_beamwidth"] = _vertical_beamwidth(row.get("Beam Scenario"))
            height = _num(pd.Series([row.get("Height")])).iloc[0]
            feats["height"] = float(height) if pd.notna(height) else None
            feats["serving_gnodeb"] = row.get("gNodeB ID")

            lat_e = _num(pd.Series([row.get("Latitude")])).iloc[0]
            lon_e = _num(pd.Series([row.get("Longitude")])).iloc[0]
            lat_d = _num(drive_df.get("Latitude")).median() if drive_df.get("Latitude") is not None else None
            lon_d = _num(drive_df.get("Longitude")).median() if drive_df.get("Longitude") is not None else None
            if pd.notna(lat_e) and pd.notna(lon_e) and lat_d is not None and pd.notna(lat_d) and lon_d is not None and pd.notna(lon_d):
                feats["distance_km"] = round(haversine_km(float(lat_e), float(lon_e), float(lat_d), float(lon_d)), 3)

        if feats["serving_gnodeb"] is not None:
            eng_indexed = eng_df.assign(_pci=eng_pci)
            noncoloc = []
            for p in feats["comparable_strength_neighbors"]:
                rows = eng_indexed[eng_indexed["_pci"] == p]
                if not rows.empty and rows.iloc[0]["gNodeB ID"] != feats["serving_gnodeb"]:
                    noncoloc.append(p)
            feats["non_colocated_overlap_pcis"] = noncoloc

    return feats


# --------------------------------------------------------------------------
# Synthetic chain-of-thought
# --------------------------------------------------------------------------

def _fmt(v, suffix=""):
    return f"{v}{suffix}" if v is not None else "an unresolved value"


def synthesize_reasoning(true_label: str, feats: dict) -> str:
    f = feats or {}
    if true_label == "C1":
        return (
            f"The serving cell's total downtilt is about {_fmt(f.get('downtilt_total'), '°')} against a vertical "
            f"beamwidth of {_fmt(f.get('vertical_beamwidth'), '°')} at a mounting height of {_fmt(f.get('height'), 'm')}, "
            f"which narrows the effective coverage radius and pulls the main beam toward the ground close to the site. "
            f"The serving SS-RSRP averages {_fmt(f.get('avg_serving_rsrp'), ' dBm')}, consistent with weak far-end "
            f"coverage caused by excessive downtilt."
        )
    if true_label == "C2":
        return (
            f"The serving cell's total downtilt of about {_fmt(f.get('downtilt_total'), '°')} is shallow relative to "
            f"its vertical beamwidth of {_fmt(f.get('vertical_beamwidth'), '°')}, extending the main beam's reach well "
            f"beyond the intended ~1km coverage radius, indicating the cell is over-shooting its intended coverage "
            f"area rather than being properly contained."
        )
    if true_label == "C3":
        return (
            f"A neighboring cell reaches this location with a filtered Tx BRSRP of about "
            f"{_fmt(f.get('best_neighbor_brsrp'), ' dBm')}, comparable to or stronger than the serving cell's average "
            f"SS-RSRP of {_fmt(f.get('avg_serving_rsrp'), ' dBm')}, suggesting a neighbor would deliver higher "
            f"throughput than the current serving cell."
        )
    if true_label == "C4":
        pcis = f.get("non_colocated_overlap_pcis") or f.get("comparable_strength_neighbors") or []
        return (
            f"Neighbor cell(s) {pcis if pcis else '[not isolated from the data]'} arrive at comparable signal strength "
            f"to the serving cell but originate from a different, non-colocated site, producing severe overlapping "
            f"coverage that degrades throughput."
        )
    if true_label == "C5":
        return (
            f"The serving PCI changes {f.get('handover_count', 'several')} times across this short drive-test window, "
            f"indicating frequent handovers that interrupt scheduling and degrade sustained throughput."
        )
    if true_label == "C6":
        collisions = f.get("pci_mod30_collisions") or []
        return (
            f"The serving cell (PCI {f.get('serving_pci', '?')}) shares PCI mod 30 with neighbor PCI(s) "
            f"{collisions if collisions else '[not isolated from the data]'}, causing reference-signal collision and "
            f"interference that suppresses throughput."
        )
    if true_label == "C7":
        return (
            f"The test vehicle's GPS speed averages {_fmt(f.get('avg_speed'), ' km/h')} and peaks at "
            f"{_fmt(f.get('max_speed'), ' km/h')}, with the peak exceeding the 40km/h mobility threshold at which "
            f"scheduling and channel estimation degrade, impacting throughput."
        )
    if true_label == "C8":
        return (
            f"The average scheduled RB count is about {_fmt(f.get('avg_rb'))}, below the 160-RB threshold needed to "
            f"sustain high throughput, directly explaining the low observed data rate."
        )
    return "Based on the provided data, this is the most consistent root cause."


# --------------------------------------------------------------------------
# Reformatting train.csv rows into test-style shuffled questions
# --------------------------------------------------------------------------

def _choose_subset_size(rng: random.Random) -> int:
    sizes = list(SUBSET_SIZE_WEIGHTS.keys())
    weights = list(SUBSET_SIZE_WEIGHTS.values())
    return rng.choices(sizes, weights=weights, k=1)[0]


def _choose_prefix_style(rng: random.Random):
    """Returns None for plain-digit labels, else a single uppercase letter
    used as a "<letter><digit>" prefix (e.g. 'M' -> 'M1', 'M2', ...)."""
    if rng.random() < PREFIX_DIGIT_PROB:
        return None
    return rng.choice(LETTERS)


def reformat_example(question_text: str, true_label: str, rng: random.Random):
    """Mimic test.csv's real distribution: a random subset of the 8 canonical
    options (always including the true label) in random order, labeled with
    a randomly chosen prefix style (plain digits or LETTER+digit)."""
    matches = list(C_OPTION_RE.finditer(question_text))
    if not matches:
        raise ValueError("no canonical C-prefixed options found in question")

    labels = [f"C{m.group(1)}" for m in matches]
    descs = [m.group(2).strip() for m in matches]
    label_to_desc = dict(zip(labels, descs))

    subset_size = _choose_subset_size(rng)
    distractors = [lbl for lbl in labels if lbl != true_label]
    chosen_distractors = rng.sample(distractors, min(subset_size - 1, len(distractors)))
    subset_labels = chosen_distractors + [true_label]
    rng.shuffle(subset_labels)

    prefix = _choose_prefix_style(rng)
    new_position = {}
    new_lines = []
    for i, lbl in enumerate(subset_labels, start=1):
        pos_str = f"{prefix}{i}" if prefix else str(i)
        new_position[lbl] = pos_str
        new_lines.append(f"{pos_str}: {label_to_desc[lbl]}")
    new_block = "\n".join(new_lines)

    start, end = matches[0].start(), matches[-1].end()
    new_question = question_text[:start] + new_block + question_text[end:]
    new_question = OPTION_COUNT_RE.sub(
        lambda m: f"following {len(subset_labels)} potential {m.group(2)}", new_question, count=1
    )

    true_position = new_position[true_label]

    try:
        drive_df = parse_drive_test_table(question_text)
        eng_df = parse_engineering_params(question_text)
        feats = compute_features(drive_df, eng_df)
    except Exception:
        feats = {}

    reasoning = synthesize_reasoning(true_label, feats)
    target_completion = f"{reasoning}\n\\boxed{{{true_position}}}"
    return new_question, target_completion


def build_sft_dataset(train_csv_path: str, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(train_csv_path)
    rng = random.Random(seed)
    rows = []
    for _, r in df.iterrows():
        true_label = str(r["answer"]).strip()
        try:
            new_q, target = reformat_example(r["question"], true_label, rng)
        except Exception as e:
            print(f"skip {r['ID']}: {e}")
            continue
        rows.append({"ID": r["ID"], "prompt": new_q, "completion": target})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    ds = build_sft_dataset(path)
    print(f"built {len(ds)} SFT examples")
    print(ds.iloc[0]["prompt"][:600])
    print("---")
    print(ds.iloc[0]["completion"])
