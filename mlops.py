"""Small CLI for a complete local lifecycle; run --help for available commands."""

import argparse
import json
from pathlib import Path

from maintenance.mlops.store import initialize, root_path, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_path())
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["bootstrap", "retrain", "status"]:
        commands.add_parser(name)
    simulate = commands.add_parser("simulate")
    simulate.add_argument("scenario", choices=["normal", "sensor_bias", "operating_shift", "schema_failure"])
    simulate.add_argument("--seed", type=int, default=2026)
    monitor = commands.add_parser("monitor")
    monitor.add_argument("batch", type=Path)
    monitor.add_argument("--unlabelled", action="store_true")
    for name in ["qualify", "promote", "rollback"]:
        action = commands.add_parser(name)
        action.add_argument("version")
        if name != "qualify":
            action.add_argument("--reason", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    initialize(root)
    try:
        if args.command == "bootstrap":
            from maintenance.mlops.lifecycle import bootstrap

            result = {"registered_version": bootstrap(root)}
        elif args.command == "retrain":
            from maintenance.mlops.lifecycle import retrain

            result = retrain(root)
        elif args.command == "qualify":
            from maintenance.mlops.lifecycle import qualify

            result = qualify(root, args.version)
        elif args.command in ["promote", "rollback"]:
            from maintenance.mlops.registry import switch

            switch(root, args.version, args.reason, rollback=args.command == "rollback")
            result = {"action": args.command, "active_version": args.version}
        elif args.command == "simulate":
            from maintenance.mlops.monitoring import simulate

            result = {"batch": str(simulate(root, args.scenario, args.seed))}
        elif args.command == "monitor":
            from maintenance.mlops.monitoring import monitor

            result = monitor(root, args.batch, labelled=not args.unlabelled)
        else:
            result = {
                "production": rows(root, "SELECT * FROM production_model"),
                "versions": rows(root, "SELECT version,lifecycle,mlflow_run_id FROM model_versions"),
                "events": rows(root, "SELECT * FROM lifecycle_events ORDER BY id DESC LIMIT 10"),
            }
        print(json.dumps(result, indent=2))
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"MLOps action stopped: {exc}\n")


if __name__ == "__main__":
    main()
