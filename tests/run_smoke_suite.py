"""Sequential GPU integration suite; outputs stay under ignored logs/."""
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    suite = root / "logs" / ("pems-adaptation-" + time.strftime("%Y%m%d-%H%M%S"))
    suite.mkdir(parents=True)
    print(f"Suite directory: {suite}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    for dataset in ("pems07", "pems08", "pems04"):
        for mode in ("tfp", "unica", "adapter_only"):
            prefix = str(suite / f"{mode}-")
            command = [sys.executable, "-B", "tests/smoke_training.py", "--data", dataset,
                       "--data_dir", f"data/{dataset}_12_12", "--input_len", "12", "--output_len", "12",
                       "--llm_layer", "6", "--U", "1", "--device", "cuda:0", "--batch_size", "1",
                       "--epochs", "1", "--max_train_batches", "2", "--max_eval_batches", "1", "--save", prefix]
            if mode != "tfp":
                command += ["--unica_enabled"]
            if mode == "adapter_only":
                command += ["--unica_training_policy", "adapter_only", "--tfp_moe_loss_weight", "0",
                            "--unica_pretrained_path", str(suite / f"tfp-{dataset}" / "best_model.pth")]
            log = suite / f"{dataset}-{mode}.log"
            print(f"Running {dataset} {mode}: {log}", flush=True)
            with log.open("w") as output:
                result = subprocess.run(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text()[-6000:], flush=True)
                raise SystemExit(result.returncode)
            print(f"PASS {dataset} {mode}", flush=True)
    print(f"All 9 integration runs passed: {suite}", flush=True)


if __name__ == "__main__":
    main()
