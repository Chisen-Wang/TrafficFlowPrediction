"""Print every experiment run next to the reference scores, ordered by validation MAE.

Usage: python experiments/summarize.py [nyc_bike|pems04|pems08]

Each run logs validation MAE every epoch, and test MAE too when it selects on
test, so both selections are shown. The validation-selected scores are the ones
a paper can report; the test-selected ones show the best the run reached.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd


LOGS = Path(__file__).resolve().parents[1] / "logs"
# Scores measured before these experiments, all selected on validation: name, val MAE, test MAE, test RMSE.
# PEMS rows are 3-seed means from the comparison server's compare.csv.
REFERENCES = {
    "nyc_bike": [
        ("reference: STAEFormer, 3 seeds", 1.6031, 1.7998, 3.5519),
        ("reference: PDFormer, 3 seeds", 1.6056, 1.8228, 3.6710),
        ("reference: STID, 3 seeds", 1.6429, 1.8411, 3.6879),
        ("reference: L=2, U=2 baseline", 1.6929, 1.8763, 3.7406),
        ("reference: level prior alone, no ML", 1.7085, 1.9010, 3.6554),
    ],
    "pems04": [
        ("reference: STD-PLM, 3 seeds", 18.3227, 18.3157, 29.9422),
        ("reference: STID, 3 seeds", 18.4779, 18.3592, 29.8408),
        ("reference: PDFormer, 3 seeds", 18.7485, 18.6545, 30.2817),
        ("reference: STAEformer, 3 seeds", 18.8322, 18.6328, 30.0919),
        ("reference: ST-LLM, 3 seeds", 21.5254, 21.2244, 32.4050),
    ],
    "pems08": [
        ("reference: STD-PLM, 3 seeds", 13.7096, 13.6354, 23.3630),
        ("reference: STAEformer, 3 seeds", 14.0252, 13.8623, 23.2212),
        ("reference: PDFormer, 3 seeds", 14.1217, 14.0241, 24.0537),
        ("reference: STID, 3 seeds", 14.4005, 14.2399, 23.3372),
        ("reference: ST-LLM, 3 seeds", 17.3314, 17.0115, 25.5650),
    ],
}
NAN = float("nan")


def experiment_rows(dataset):
    for run in sorted(LOGS.glob(f"*-seed*-{dataset}")):
        path = run / "train.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path, index_col=0).reset_index(drop=True)
        patience = json.loads((run / "run_config.json").read_text())["es_patience"]
        best = int(frame.valid_loss.idxmin())
        row = dict(name=run.name.removesuffix(f"-{dataset}"), epochs=len(frame), val_epoch=best + 1,
                   val=frame.valid_loss[best], test=NAN, rmse=NAN, test_epoch=None, test_best=NAN,
                   # A validation-driven run would stop `patience` epochs after its best.
                   settled=len(frame) >= best + 1 + patience)
        if "test_loss" in frame:
            test_best = int(frame.test_loss.idxmin())
            row.update(test=frame.test_loss[best], rmse=frame.test_rmse[best],
                       test_epoch=test_best + 1, test_best=frame.test_loss[test_best])
        elif (run / "test.csv").exists():
            final = pd.read_csv(run / "test.csv", index_col=0).iloc[-1]
            row.update(test=final.test_loss, rmse=final.test_rmse)
        yield row


def number(value, digits=4):
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "nyc_bike"
    if dataset not in REFERENCES:
        raise SystemExit(f"Expected one of {', '.join(REFERENCES)}, got {dataset}")
    rows = list(experiment_rows(dataset))
    rows += [dict(name=name, epochs=None, val_epoch=None, val=val, test=test, rmse=rmse, test_epoch=None,
                  test_best=NAN, settled=True) for name, val, test, rmse in REFERENCES[dataset]]
    rows.sort(key=lambda row: row["val"])
    width = max(len(row["name"]) for row in rows)
    print(f"{'run':{width}}  {'epochs':>6}  | selected on validation: {'epoch':>5} {'val MAE':>8} {'test MAE':>8} {'RMSE':>7}"
          f"  | selected on test: {'epoch':>5} {'test MAE':>8}")
    for row in rows:
        note = "" if row["settled"] else "  (a validation-selected run would have trained longer)"
        print(f"{row['name']:{width}}  {row['epochs'] or '-':>6}  | {'':24}{row['val_epoch'] or '-':>5} {number(row['val']):>8} "
              f"{number(row['test']):>8} {number(row['rmse']):>7}  | {'':18}{row['test_epoch'] or '-':>5} "
              f"{number(row['test_best']):>8}{note}")


if __name__ == "__main__":
    main()
