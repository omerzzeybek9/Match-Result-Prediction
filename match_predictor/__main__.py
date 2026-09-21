import argparse
import json
from datetime import date
from pathlib import Path

from .data import LEAGUES, ROOT


def main():
    parser = argparse.ArgumentParser(description="Football forecasting: download, train, predict")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--leagues", nargs="+", choices=LEAGUES, default=[x for x in LEAGUES if x != "cl"])
    download.add_argument("--start", type=int, default=2018)
    download.add_argument("--end", type=int, default=date.today().year - (date.today().month < 7))
    download.add_argument("--data-dir", type=Path)
    train = commands.add_parser("train")
    train.add_argument("--leagues", nargs="+", choices=LEAGUES, default=[x for x in LEAGUES if x != "cl"])
    train.add_argument("--data-dir", type=Path)
    train.add_argument("--output-dir", type=Path, default=ROOT/"artifacts")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--league", choices=LEAGUES, required=True)
    prediction.add_argument("--home", required=True)
    prediction.add_argument("--away", required=True)
    prediction.add_argument("--date")
    prediction.add_argument("--model-dir", type=Path, default=ROOT/"artifacts")
    report = commands.add_parser("report")
    report.add_argument("--model-dir", type=Path, default=ROOT/"artifacts")
    report.add_argument("--output", type=Path, default=ROOT/"reports/benchmark.md")
    args = parser.parse_args()
    if args.command == "download":
        from .data import download_history
        errors = download_history(args.leagues, args.start, args.end, args.data_dir)
        if errors:
            raise SystemExit("Some seasons failed; see messages above. Existing files were retained.")
    elif args.command == "train":
        from threadpoolctl import threadpool_limits
        from .training import train_league
        # Avoid excessive OpenMP/BLAS thread creation on laptops and hosted runners.
        with threadpool_limits(limits=1):
            for league in args.leagues:
                train_league(league, args.output_dir, args.data_dir)
    elif args.command == "report":
        from .reporting import write_benchmark
        print(write_benchmark(args.model_dir, args.output))
    else:
        import joblib
        from .predict import predict_match
        path = args.model_dir/f"{args.league}.joblib"
        if not path.exists():
            parser.error(f"Missing {path}; run the train command first")
        bundle = joblib.load(path)
        print(json.dumps(predict_match(bundle,args.home,args.away,args.date),indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
