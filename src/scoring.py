"""Answer extraction and Pass@1 scoring against validation_target.csv.

Pure regex/pandas — no GPU or model needed.
"""
import re

import pandas as pd

from data_prep import CANONICAL_DESCRIPTIONS, parse_options

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed(text: str):
    if not isinstance(text, str):
        return None
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def map_position_to_label(question_text: str, canonical_descriptions=None):
    """{position_str: canonical_label} for a shuffled test/validation question,
    by matching each option's description text against the known canonical wording.
    Falls back to identity mapping (position -> position) for questions whose
    options don't match any canonical description (e.g. generic knowledge MCQs,
    where the 'label' is just its own position number).
    """
    canonical_descriptions = canonical_descriptions or CANONICAL_DESCRIPTIONS
    desc_to_label = {desc.strip(): label for label, desc in canonical_descriptions.items()}
    options = parse_options(question_text)
    mapping = {}
    for pos, desc in options:
        mapping[pos] = desc_to_label.get(desc.strip(), pos)
    return mapping


def pass_at_1(pred_df: pd.DataFrame, target_df: pd.DataFrame, question_lookup: dict,
              canonical_descriptions=None) -> float:
    """pred_df: columns ID (suffixed _1.._4), Target (generated text containing \\boxed{N})
    target_df: columns ID (suffixed _1.._4), Target (ground-truth canonical label, e.g. 'C2')
    question_lookup: {base_id (no suffix): question_text}
    """
    merged = pred_df.merge(target_df, on="ID", suffixes=("_pred", "_true"))
    correct, total = 0, 0
    for _, row in merged.iterrows():
        base_id = row["ID"].rsplit("_", 1)[0]
        qtext = question_lookup.get(base_id)
        if qtext is None:
            continue
        pos_map = map_position_to_label(qtext, canonical_descriptions)
        boxed = extract_boxed(row["Target_pred"])
        pred_label = pos_map.get(boxed)
        total += 1
        if pred_label == row["Target_true"]:
            correct += 1
    return correct / total if total else 0.0


if __name__ == "__main__":
    import sys

    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    vq = pd.read_csv(f"{base_dir}/validation_questions.csv")
    vt = pd.read_csv(f"{base_dir}/validation_target.csv")
    lookup = dict(zip(vq["ID"], vq["question"]))

    # Fake "predictions" that just parrot the correct answer, to sanity-check
    # the extraction/mapping/scoring plumbing end-to-end without a model.
    fake_rows = []
    for _, row in vt.iterrows():
        base_id = row["ID"].rsplit("_", 1)[0]
        qtext = lookup[base_id]
        pos_map = map_position_to_label(qtext)
        true_label = row["Target"]
        true_positions = [p for p, lbl in pos_map.items() if lbl == true_label]
        pos = true_positions[0] if true_positions else "0"
        fake_rows.append({"ID": row["ID"], "Target": f"dummy reasoning \\boxed{{{pos}}}"})
    fake_pred = pd.DataFrame(fake_rows)

    score = pass_at_1(fake_pred, vt, lookup)
    print(f"sanity pass@1 with oracle predictions (should be ~1.0): {score:.4f}")
