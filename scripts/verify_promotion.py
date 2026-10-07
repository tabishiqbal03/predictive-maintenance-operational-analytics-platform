"""Demonstrate qualified promotion and rollback against one live API process."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx

from maintenance.mlops.registry import active, switch
from maintenance.mlops.store import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    args = parser.parse_args()
    original = active(ROOT)["version"]
    if original == args.candidate:
        parser.error("Candidate must differ from active production")
    payload = json.loads((ROOT / "artifacts/example_request.json").read_text())
    result = {"previous_version": original, "candidate_version": args.candidate}
    with (ROOT / "artifacts/mlops/promotion_server.log").open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8768"],
            cwd=ROOT,
            stdout=output,
            stderr=output,
        )
        try:
            with httpx.Client(base_url="http://127.0.0.1:8768", timeout=30) as api:
                for _ in range(60):
                    try:
                        response = api.get("/health")
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError("API startup timed out")
                before = api.post("/predict", json=payload).json()
                assert before["model_version"] == original
                switch(ROOT, args.candidate, "Qualified local lifecycle demonstration")
                promoted = api.post("/predict", json=payload).json()
                assert promoted["model_version"] == args.candidate
                switch(
                    ROOT, original, "Verified rollback: restore original benchmark champion", rollback=True
                )
                restored = api.post("/predict", json=payload).json()
                assert restored["model_version"] == original
                assert abs(restored["predicted_rul"] - before["predicted_rul"]) < 1e-10
                result.update(
                    promotion="passed",
                    rollback="passed",
                    same_api_process=True,
                    before=before,
                    promoted=promoted,
                    restored=restored,
                )
        finally:
            if active(ROOT)["version"] != original:
                switch(ROOT, original, "Restore baseline after lifecycle verification", rollback=True)
            process.terminate()
            process.wait(timeout=15)
    write_json(ROOT / "artifacts/mlops/promotion_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
