from typing import List, Dict, Optional

import pandas as pd
import json
import logging
from itertools import chain, combinations

logging.basicConfig(level=logging.INFO)

def extract_metadata(df):
    try:
        metadata = {
            "forecast_method": df.iloc[0]['Method'],
            "parameters": df.iloc[0]['Parameters'],
            "ufm_id": df.iloc[0]['UserForecastMethodID'],
            "start_date": df.iloc[0]['StartDate'],
            "end_date": df.iloc[0]['EndDate'],
            "databrick_id": df.iloc[0]['DatabrickID'],
        }
        return metadata
    except (KeyError, IndexError) as e:
        logging.error(f"Failed to extract metadata: {e}")
        raise


def parse_json_column(df, column_name, key=None):
    if column_name not in df.columns:
        logging.warning(f"Column '{column_name}' does not exist.")
        return []

    extracted = []
    for row in df[column_name].dropna():
        try:
            parsed = json.loads(row)
            if key:
                value = parsed.get(key, [])
            else:
                value = [v for entry in parsed.items() for v in entry[1]] if isinstance(parsed, dict) else []
            extracted.extend(value)
        except json.JSONDecodeError as e:
            logging.warning(f"Skipping invalid JSON: {e}")
            continue

    return list(set([v for v in extracted if v]))

def generate_combinations(columns=None) -> Dict[frozenset, List[str]]:
    """
    Generate all non-empty combinations of prediction columns, mapping each to a reporting structure.
    """
    if columns is None:
        columns = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
                   "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"]
    combo_map = {
        frozenset(c): ['ReportingMonth', 'CustomerID'] + list(c)
        for r in range(1, len(columns) + 1)
        for c in combinations(columns, r)
    }
    logging.info(f"Generated {len(combo_map)} column combinations.")
    # type help(generate_combinations)
    return combo_map


def find_matching_combination(combos: Dict[frozenset, List[str]], target_columns = None ) -> Optional[List[str]]:
    """
    Find the best matching key from combinations mapping.
    Tries for exact/full set match first.
    """
    if target_columns is None:
        target_columns = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
                   "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"]


    target_set = frozenset(target_columns)

    for key in combos.keys():
        if key == target_set:
            logging.info(f"Exact match found for: {target_set}")
            return combos[key]

    logging.warning(f"No exact match found for: {target_set}")
    return None