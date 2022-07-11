import json
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd


def read_lm_kbc_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Reads a LM-KBC jsonl file and returns a list of dictionaries.

    Args:
        file_path: path to the jsonl file

    Returns:
        list of dictionaries, each possibly has the following keys:
        - "SubjectEntity": str
        - "Relation": str
        - "ObjectEntities":
            None or List[List[str]] (can be omitted for the test input)
    """
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    return rows


def read_lm_kbc_jsonl_to_df(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a LM-KBC jsonl file and returns a dataframe.
    """
    rows = read_lm_kbc_jsonl(file_path)
    df = pd.DataFrame(rows)
    return df
