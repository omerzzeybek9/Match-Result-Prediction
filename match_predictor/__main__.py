import argparse
import json
from datetime import date, datetime, timezone
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
    prediction.add_argument("--home-odds", type=float)
    prediction.add_argument("--draw-odds", type=float)
    prediction.add_argument("--away-odds", type=float)
    prediction.add_argument("--model-dir", type=Path, default=ROOT/"artifacts")
    report = commands.add_parser("report")
    report.add_argument("--model-dir", type=Path, default=ROOT/"artifacts")
    report.add_argument("--output", type=Path, default=ROOT/"reports/benchmark.md")
    collect = commands.add_parser("collect", help="Capture pre-match API-Football context snapshots")
    collect.add_argument("--fixture", type=int, action="append",
                         help="API-Football fixture ID; repeat for several fixtures")
    collect.add_argument("--league", choices=[name for name in LEAGUES if name != "cl"],
                         help="Repository league key used to discover fixtures")
    collect.add_argument("--season", type=int,
                         help="API-Football season start year, for example 2026")
    collect.add_argument("--from-date", help="Fixture discovery window start, YYYY-MM-DD")
    collect.add_argument("--to-date", help="Fixture discovery window end, YYYY-MM-DD")
    collect.add_argument("--details", action="store_true",
                         help="After discovery, fetch lineups, injuries and odds for every fixture")
    collect.add_argument("--without-odds", action="store_true",
                         help="Skip the odds endpoint when capturing fixture details")
    collect.add_argument("--output-dir", type=Path, default=ROOT/"data/api_football")
    args = parser.parse_args()
    if args.command == "download":
        from .data import download_history
        errors = download_history(args.leagues, args.start, args.end, args.data_dir)
        if errors:
            raise SystemExit("Some seasons failed; see messages above. Existing files were retained.")
    elif args.command == "train":
        from threadpoolctl import threadpool_limits
        from .training import finalize_selective_policies, train_league
        # Avoid excessive OpenMP/BLAS thread creation on laptops and hosted runners.
        with threadpool_limits(limits=1):
            for league in args.leagues:
                train_league(league, args.output_dir, args.data_dir)
            finalize_selective_policies(args.output_dir,args.leagues)
    elif args.command == "report":
        from .reporting import write_benchmark
        print(write_benchmark(args.model_dir, args.output))
    elif args.command == "collect":
        from .api_football import ApiFootballClient, SnapshotStore, collect_fixture
        if not args.fixture and not (args.league and args.season):
            parser.error("Use --fixture or both --league and --season")
        if args.league and args.season is None:
            parser.error("--season is required when --league is used")
        try:
            client = ApiFootballClient.from_environment()
        except ValueError as error:
            parser.error(str(error))
        store = SnapshotStore(args.output_dir)
        fixture_ids = list(dict.fromkeys(args.fixture or []))
        if args.league:
            listing = client.fixtures(args.league, args.season, args.from_date, args.to_date)
            listing_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            listing_path = store.save_listing(listing, listing_time)
            discovered = []
            for item in listing.get("response", []):
                if isinstance(item, dict) and isinstance(item.get("fixture"), dict):
                    value = item["fixture"].get("id")
                    if value is not None:
                        discovered.append(int(value))
            fixture_ids = list(dict.fromkeys(fixture_ids + discovered))
            print(f"Saved fixture list: {listing_path} ({len(discovered)} fixtures)", flush=True)
            if not args.details and not args.fixture:
                print("Discovery only. Re-run with --details to fetch lineups, injuries and odds.", flush=True)
        if args.details or args.fixture:
            for fixture_id in fixture_ids:
                path, record = collect_fixture(client, fixture_id, store, include_odds=not args.without_odds)
                normalized = store.save_normalized(record)
                print(json.dumps({"fixture_id": fixture_id, "raw": str(path),
                                  "normalized": str(normalized),
                                  "lineups_confirmed": record["lineups"]["home"]["confirmed"] and record["lineups"]["away"]["confirmed"],
                                  "injuries": record["injuries"]["home_count"] + record["injuries"]["away_count"],
                                  "odds_ready": record["odds"]["normalized_probability"] is not None}, ensure_ascii=False), flush=True)
    else:
        import joblib
        from .predict import predict_match
        path = args.model_dir/f"{args.league}.joblib"
        if not path.exists():
            parser.error(f"Missing {path}; run the train command first")
        bundle = joblib.load(path)
        odds=[args.home_odds,args.draw_odds,args.away_odds]
        if any(value is not None for value in odds) and not all(value is not None for value in odds):
            parser.error("Provide all three odds or none")
        print(json.dumps(predict_match(bundle,args.home,args.away,args.date,
              odds if all(value is not None for value in odds) else None),indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
