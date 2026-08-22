REASONABLE_LABEL = "合理"

LABEL_OPTION_MAP = {
    "A": REASONABLE_LABEL,
    "B": "用法、用量不适宜",
    "C": "适应症不适宜",
    "D": "药品剂型或给药途径不适宜",
    "E": "有配伍禁忌或不良相互作用",
    "F": "重复给药",
    "G": "遴选的药品不适宜",
}

LABEL_CANONICAL_MAP = {
    REASONABLE_LABEL: REASONABLE_LABEL,
    "A": REASONABLE_LABEL,
    "B": LABEL_OPTION_MAP["B"],
    "C": LABEL_OPTION_MAP["C"],
    "D": LABEL_OPTION_MAP["D"],
    "E": LABEL_OPTION_MAP["E"],
    "F": LABEL_OPTION_MAP["F"],
    "G": LABEL_OPTION_MAP["G"],
    "a": REASONABLE_LABEL,
    "b": LABEL_OPTION_MAP["B"],
    "c": LABEL_OPTION_MAP["C"],
    "d": LABEL_OPTION_MAP["D"],
    "e": LABEL_OPTION_MAP["E"],
    "f": LABEL_OPTION_MAP["F"],
    "g": LABEL_OPTION_MAP["G"],
    "用法、用量不适宜": LABEL_OPTION_MAP["B"],
    "用法用量不适宜": LABEL_OPTION_MAP["B"],
    "适应症不适宜": LABEL_OPTION_MAP["C"],
    "药品剂型或给药途径不适宜": LABEL_OPTION_MAP["D"],
    "给药途径不适宜": LABEL_OPTION_MAP["D"],
    "剂型不适宜": LABEL_OPTION_MAP["D"],
    "有配伍禁忌或不良相互作用": LABEL_OPTION_MAP["E"],
    "有配伍禁忌或者不良相互作用": LABEL_OPTION_MAP["E"],
    "有配伍禁忌或不良相互作用风险": LABEL_OPTION_MAP["E"],
    "重复给药": LABEL_OPTION_MAP["F"],
    "遴选的药品不适宜": LABEL_OPTION_MAP["G"],
    "遴选药品不适宜": LABEL_OPTION_MAP["G"],
    "适用人群不适宜": LABEL_OPTION_MAP["G"],
}

ALL_ERROR_LABELS = [
    LABEL_OPTION_MAP["B"],
    LABEL_OPTION_MAP["C"],
    LABEL_OPTION_MAP["D"],
    LABEL_OPTION_MAP["E"],
    LABEL_OPTION_MAP["F"],
    LABEL_OPTION_MAP["G"],
]

RELATION_LABELS = {
    LABEL_OPTION_MAP["E"],
    LABEL_OPTION_MAP["F"],
}

OPTION_LABELS = {key: value for key, value in LABEL_OPTION_MAP.items()}
OPTION_BY_LABEL = {value: key for key, value in LABEL_OPTION_MAP.items()}


def canonicalize_label(label: str) -> str:
    value = str(label or "").strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    return LABEL_CANONICAL_MAP.get(value, value)


def label_to_option(label: str) -> str:
    return OPTION_BY_LABEL.get(canonicalize_label(label), "")


def labels_to_option_string(is_reasonable: bool, labels) -> str:
    if bool(is_reasonable):
        return "A"
    options = sorted({label_to_option(label) for label in (labels or []) if label_to_option(label)})
    return ",".join(options) if options else "UNKNOWN"
