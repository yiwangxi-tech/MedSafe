import json
import os
from typing import Any, Dict, List

import pandas as pd


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv_row(path: str, row: Dict[str, Any]):
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        index=False,
        header=not os.path.exists(path),
        encoding="utf-8-sig",
    )


def load_progress(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame() if df.empty else df


def loads_list(value: Any) -> List[Any]:
    if pd.isna(value) or value == "":
        return []
    return json.loads(value)
