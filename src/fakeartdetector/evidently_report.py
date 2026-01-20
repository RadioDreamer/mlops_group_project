import base64
import os
from sqlite3 import connect

import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report


def normalize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    def normalize(x):
        if isinstance(x, (bytes, bytearray)):
            # encode binary embedding safely
            return base64.b64encode(x).decode("ascii")
        if isinstance(x, memoryview):
            return base64.b64encode(x.tobytes()).decode("ascii")
        if isinstance(x, np.generic):
            return x.item()
        return x

    return df.map(normalize)


def generate_evidently_report(db_path: str, out_html: str, reference_n: int = 1000, current_n: int = 200):
    conn = connect(db_path)
    df = pd.read_sql_query("SELECT * FROM inference_logs ORDER BY timestamp", conn)
    if df.empty:
        raise RuntimeError("No rows in inference_logs")
    df = normalize_for_json(df)

    ref = df.head(reference_n)
    cur = df.tail(current_n)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    report.save_html(out_html)
    return out_html


if __name__ == "__main__":
    out = generate_evidently_report(
        db_path="data/inference_logs/inference_logs.db",
        out_html="data/evidently_reports/evidently_report.html",
    )
