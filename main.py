# -*- coding: utf-8 -*-
"""
keirin-main CLI entry point
競輪データの取得・学習・予測・買い目生成を行うツール
"""

import argparse
from pathlib import Path
from utils import get_logger, ensure_directory
from schedule.providers import list_race_ids_for_date
from scrape.gamboo import race_data_scrape
from scrape.normalize import to_training_csv, to_cards_csv
from model import Model
from bets import simulate_bets
from datetime import datetime
import pandas as pd

logger = get_logger(__name__)


# =============================
# サブコマンド: fetch
# =============================
def cmd_fetch(args: argparse.Namespace) -> None:
    if args.race_ids:
        race_ids = args.race_ids.split(",")
    elif args.date:
        race_ids = list_race_ids_for_date(args.date, providers=args.providers)
    else:
        logger.error("日付またはレースIDを指定してください")
        return

    info_df, entry_df, payout_df = race_data_scrape(
        race_ids,
        rate_limit=args.rate_limit,
        retries=args.retries,
        timeout=args.timeout,
    )

    out_dir = ensure_directory(args.out)
    to_training_csv(entry_df, info_df, payout_df, str(out_dir / "races.csv"))
    to_cards_csv(entry_df, str(out_dir / "cards.csv"))


# =============================
# サブコマンド: train
# =============================
def cmd_train(args: argparse.Namespace) -> None:
    races_path = Path(args.races)
    if not races_path.exists():
        logger.error("学習用ファイルが存在しません: %s", races_path)
        return

    df = pd.read_csv(races_path)
    model = Model()
    model.train(df)
    model.save(args.out)
    logger.info("モデルを保存しました: %s", args.out)


# =============================
# サブコマンド: predict
# =============================
def cmd_predict(args: argparse.Namespace) -> None:
    model = Model.load(args.model)
    df = pd.read_csv(args.cards)
    preds = model.predict(df)
    preds.to_csv(args.out, index=False)
    logger.info("予測結果を保存しました: %s", args.out)


# =============================
# サブコマンド: backtest
# =============================
def cmd_backtest(args: argparse.Namespace) -> None:
    model = Model.load(args.model)
    races = pd.read_csv(args.races)
    preds = model.predict(races)
    simulate_bets(preds, budget=args.budget, policy=args.bet_policy)
    logger.info("バックテスト完了")


# =============================
# サブコマンド: today
# =============================
def cmd_today(args: argparse.Namespace) -> None:
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    race_ids = list_race_ids_for_date(date_str, providers=args.providers)
    if not race_ids:
        logger.warning("対象レースが見つかりませんでした")
        return

    info_df, entry_df, payout_df = race_data_scrape(
        race_ids,
        rate_limit=args.rate_limit,
        retries=args.retries,
        timeout=args.timeout,
    )

    tmp_dir = ensure_directory(Path(args.out) / date_str)
    to_training_csv(entry_df, info_df, payout_df, str(tmp_dir / "races.csv"))
    to_cards_csv(entry_df, str(tmp_dir / "cards.csv"))

    # 予測
    model = Model.load(args.model)
    df_cards = pd.read_csv(tmp_dir / "cards.csv")
    preds = model.predict(df_cards)
    preds.to_csv(tmp_dir / f"predictions_{date_str}.csv", index=False)

    # 買い目
    bets = simulate_bets(preds, budget=args.budget, policy=args.bet_policy)
    bets.to_csv(tmp_dir / f"today_bets_{date_str}.csv", index=False)
    logger.info("今日の買い目を出力しました: %s", tmp_dir)


# =============================
# CLI パーサー設定
# =============================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="競輪予測ツール CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- fetch ---
    fetch_parser = subparsers.add_parser("fetch", help="レース情報を取得")
    fetch_parser.add_argument("--race-ids", help="取得するレースID（カンマ区切り）")
    fetch_parser.add_argument("--date", help="取得日 (YYYY-MM-DD)")
    fetch_parser.add_argument("--providers", default="chariloto,kdreams,keirin_jp")
    fetch_parser.add_argument("--out", required=True, help="出力ディレクトリ")
    fetch_parser.add_argument("--rate-limit", type=float, default=1.0)
    fetch_parser.add_argument("--retries", type=int, default=3)
    fetch_parser.add_argument("--timeout", type=int, default=15)
    fetch_parser.set_defaults(func=cmd_fetch)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="モデルを学習して保存します")
    train_parser.add_argument("--races", required=True, help="学習データ（races.csv）のパス")
    train_parser.add_argument("--out", required=True, help="出力ディレクトリ")
    train_parser.set_defaults(func=cmd_train)

    # --- predict ---
    predict_parser = subparsers.add_parser("predict", help="予測を実行します")
    predict_parser.add_argument("--cards", required=True, help="出走表データ（cards.csv）のパス")
    predict_parser.add_argument("--model", required=True, help="モデルディレクトリ")
    predict_parser.add_argument("--out", required=True, help="出力ファイル名")
    predict_parser.add_argument("--ct", default="independent")
    predict_parser.set_defaults(func=cmd_predict)

    # --- backtest ---
    backtest_parser = subparsers.add_parser("backtest", help="バックテストを実行します")
    backtest_parser.add_argument("--races", required=True)
    backtest_parser.add_argument("--model", required=True)
    backtest_parser.add_argument("--budget", type=int, default=10000)
    backtest_parser.add_argument("--bet-policy", default="flat")
    backtest_parser.add_argument("--zone-filter", default="any")
    backtest_parser.set_defaults(func=cmd_backtest)

    # --- today ---
    today_parser = subparsers.add_parser("today", help="今日の買い目を自動生成")
    today_parser.add_argument("--model", required=True, help="モデルディレクトリ")
    today_parser.add_argument("--date", help="日付指定 (YYYY-MM-DD)")
    today_parser.add_argument("--budget", type=int, default=10000)
    today_parser.add_argument("--bet-policy", default="flat")
    today_parser.add_argument("--ct", default="mc")
    today_parser.add_argument("--mc-iters", type=int, default=2000)
    today_parser.add_argument("--out", required=True, help="出力ディレクトリ")
    today_parser.add_argument("--providers", default="chariloto,kdreams,keirin_jp")
    today_parser.add_argument("--rate-limit", type=float, default=1.0)
    today_parser.add_argument("--retries", type=int, default=3)
    today_parser.add_argument("--timeout", type=int, default=15)
    today_parser.set_defaults(func=cmd_today)

    return parser


# =============================
# メインエントリーポイント
# =============================
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
