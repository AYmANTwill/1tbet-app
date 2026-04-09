"""
validate.py - Validation walk-forward multi-saisons

TEST DE VERITE : est-ce que la strategie Away est profitable
sur PLUSIEURS saisons, pas juste 2024-2025 ?

Methode walk-forward :
  Train: 2020-2021         -> Test: 2021-2022
  Train: 2020-2021,2021-22 -> Test: 2022-2023
  Train: 2020-2022,2022-23 -> Test: 2023-2024
  Train: 2020-2023,2023-24 -> Test: 2024-2025

Si le ROI est positif sur 3+ saisons, le signal est reel.
Si c'est positif sur 1 seule, c'est probablement du hasard.

Usage :
    python ml/validate.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def load_raw_data():
    """Charge les CSV bruts."""
    raw_dir = Path("data/raw")
    all_dfs = []
    for csv_file in sorted(raw_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, encoding="latin-1")
            parts = csv_file.stem.split("_")
            if len(parts) >= 2:
                df["league_code"] = parts[0]
                df["season"] = "_".join(parts[1:])
            all_dfs.append(df)
        except:
            pass
    return pd.concat(all_dfs, ignore_index=True)


def build_features(df):
    """Feature engineering (identique a research.py)."""
    df = df.copy()
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    if not all(c in df.columns for c in required):
        return None, None

    df = df.rename(columns={
        "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    })
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "result"])
    df = df.sort_values("date").reset_index(drop=True)

    # Cotes Bet365
    for col in ["B365H", "B365D", "B365A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "B365H" in df.columns:
        df["b365_home"] = df["B365H"]
        df["b365_draw"] = df["B365D"]
        df["b365_away"] = df["B365A"]
        imp_h = 1 / df["b365_home"]
        imp_d = 1 / df["b365_draw"]
        imp_a = 1 / df["b365_away"]
        total = imp_h + imp_d + imp_a
        df["book_prob_home"] = imp_h / total
        df["book_prob_draw"] = imp_d / total
        df["book_prob_away"] = imp_a / total

    # Elo
    elo = {}
    K, HA = 25, 80
    elo_probs, elo_diffs, elo_close = [], [], []
    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hr = elo.get(ht, 1500) + HA
        ar = elo.get(at, 1500)
        exp_h = 1 / (1 + 10 ** ((ar - hr) / 400))
        diff = elo.get(ht, 1500) - elo.get(at, 1500)
        elo_probs.append(exp_h)
        elo_diffs.append(diff)
        elo_close.append(1 / (1 + abs(diff) / 100))
        sh = 1.0 if row["result"] == "H" else (0.5 if row["result"] == "D" else 0.0)
        elo[ht] = elo.get(ht, 1500) + K * (sh - exp_h)
        elo[at] = elo.get(at, 1500) + K * ((1 - sh) - (1 - exp_h))
    df["elo_prob"] = elo_probs
    df["elo_diff"] = elo_diffs
    df["elo_closeness"] = elo_close

    # Forme
    for n in [3, 5]:
        df[f"home_form_{n}"] = df.groupby("home_team")["home_goals"].transform(
            lambda x: x.rolling(n, min_periods=1).mean())
        df[f"away_form_{n}"] = df.groupby("away_team")["away_goals"].transform(
            lambda x: x.rolling(n, min_periods=1).mean())
        df[f"home_form_{n}"] = df.groupby("home_team")[f"home_form_{n}"].shift(1)
        df[f"away_form_{n}"] = df.groupby("away_team")[f"away_form_{n}"].shift(1)

    df["home_def_form"] = df.groupby("home_team")["away_goals"].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df["away_def_form"] = df.groupby("away_team")["home_goals"].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df["home_def_form"] = df.groupby("home_team")["home_def_form"].shift(1)
    df["away_def_form"] = df.groupby("away_team")["away_def_form"].shift(1)

    df["goal_diff_form"] = df["home_form_5"].fillna(0) - df["away_form_5"].fillna(0)
    df["def_diff_form"] = df["away_def_form"].fillna(0) - df["home_def_form"].fillna(0)

    # Tirs
    if "HS" in df.columns:
        df["HST"] = pd.to_numeric(df.get("HST", pd.Series(dtype=float)), errors="coerce")
        df["AST"] = pd.to_numeric(df.get("AST", pd.Series(dtype=float)), errors="coerce")
        df["home_shots_on"] = df.groupby("home_team")["HST"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["away_shots_on"] = df.groupby("away_team")["AST"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["home_shots_on"] = df.groupby("home_team")["home_shots_on"].shift(1)
        df["away_shots_on"] = df.groupby("away_team")["away_shots_on"].shift(1)
        total_shots = df["home_shots_on"].fillna(0) + df["away_shots_on"].fillna(0)
        df["shot_dominance"] = np.where(total_shots > 0,
            df["home_shots_on"].fillna(0) / total_shots, 0.5)

    if "HC" in df.columns:
        df["HC"] = pd.to_numeric(df["HC"], errors="coerce")
        df["home_corners"] = df.groupby("home_team")["HC"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["home_corners"] = df.groupby("home_team")["home_corners"].shift(1)

    # Draw rate
    df["is_draw"] = (df["result"] == "D").astype(float)
    df["avg_draw_rate"] = (
        df.groupby("home_team")["is_draw"].transform(
            lambda x: x.rolling(10, min_periods=3).mean()).fillna(0.25) +
        df.groupby("away_team")["is_draw"].transform(
            lambda x: x.rolling(10, min_periods=3).mean()).fillna(0.25)
    ) / 2
    df["avg_draw_rate"] = df.groupby("home_team")["avg_draw_rate"].shift(1)

    # Target
    df["target"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    # Feature cols
    feature_cols = [
        "elo_prob", "elo_diff", "elo_closeness",
        "home_form_3", "home_form_5", "away_form_3", "away_form_5",
        "home_def_form", "away_def_form", "goal_diff_form", "def_diff_form",
        "avg_draw_rate",
    ]
    if "home_shots_on" in df.columns:
        feature_cols.extend(["home_shots_on", "away_shots_on", "shot_dominance"])
    if "home_corners" in df.columns:
        feature_cols.append("home_corners")

    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    return df, feature_cols


def backtest_away(test_df, probs, min_edge=0.05, min_odds=2.50, exclude_leagues=None):
    """Backtest Away avec filtres."""
    stake = 10.0
    bets = []
    exclude = exclude_leagues or []

    for i, (_, match) in enumerate(test_df.iterrows()):
        if match.get("league_code") in exclude:
            continue
        odds_a = match.get("b365_away")
        if pd.isna(odds_a) or odds_a < min_odds:
            continue
        edge = probs[i, 2] - 1 / odds_a
        if edge >= min_edge:
            won = match["result"] == "A"
            bets.append({
                "won": won,
                "profit": stake * (odds_a - 1) if won else -stake,
                "odds": odds_a,
                "edge": edge,
                "league": match.get("league_code", ""),
            })

    if not bets:
        return {"n": 0, "roi": 0, "profit": 0, "win_rate": 0}

    bdf = pd.DataFrame(bets)
    return {
        "n": len(bdf),
        "roi": bdf["profit"].sum() / (len(bdf) * stake) * 100,
        "profit": bdf["profit"].sum(),
        "win_rate": bdf["won"].mean(),
        "avg_odds": bdf["odds"].mean(),
        "avg_edge": bdf["edge"].mean(),
    }


def walk_forward_validation():
    """Validation walk-forward multi-saisons."""
    print("=" * 60)
    print("VALIDATION WALK-FORWARD MULTI-SAISONS")
    print("=" * 60)

    print("\n   Chargement et feature engineering...")
    df = load_raw_data()
    result = build_features(df)
    if result is None:
        print("   Erreur de chargement")
        return
    df, feature_cols = result

    seasons = sorted(df["season"].unique())
    print(f"   Saisons disponibles : {seasons}")
    print(f"   Features : {len(feature_cols)}")

    # ============================================================
    # WALK-FORWARD : train sur le passe, test sur la saison suivante
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1 : AWAY (edge>5%, cotes>2.50)")
    print("=" * 60)

    print(f"\n   {'Test season':<15} {'Train':>6} {'Test':>6} {'Paris':>6} {'Win%':>7} {'ROI':>8} {'Profit':>10}")
    print(f"   {'-'*62}")

    total_bets = 0
    total_profit = 0
    season_results = []

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_df = df[df["season"].isin(train_seasons)]
        test_df = df[df["season"] == test_season]

        if len(train_df) < 200 or len(test_df) < 50:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_test = test_df[feature_cols]

        # Entrainer
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train)

        cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
        cal.fit(X_train, y_train)
        probs = cal.predict_proba(X_test)

        # Backtest Away
        r = backtest_away(test_df, probs, min_edge=0.05, min_odds=2.50)

        marker = " $$$" if r["roi"] > 0 else ""
        if r["n"] > 0:
            print(f"   {test_season:<15} {len(train_df):>6} {len(test_df):>6} "
                  f"{r['n']:>6} {r['win_rate']:>6.1%} {r['roi']:>+7.1f}% "
                  f"{r['profit']:>+9.1f}{marker}")
            total_bets += r["n"]
            total_profit += r["profit"]
            season_results.append(r)

    if total_bets > 0:
        overall_roi = total_profit / (total_bets * 10) * 100
        profitable_seasons = sum(1 for r in season_results if r["roi"] > 0)
        print(f"   {'-'*62}")
        print(f"   {'TOTAL':<15} {'':>6} {'':>6} {total_bets:>6} "
              f"{'':>7} {overall_roi:>+7.1f}% {total_profit:>+9.1f}")
        print(f"\n   Saisons profitables : {profitable_seasons}/{len(season_results)}")

    # ============================================================
    # TEST 2 : Variantes de parametres
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2 : GRILLE DE PARAMETRES (toutes saisons combinees)")
    print("=" * 60)

    configs = [
        {"label": "Away e>3% c>2.0", "min_edge": 0.03, "min_odds": 2.0, "exclude": []},
        {"label": "Away e>5% c>2.5", "min_edge": 0.05, "min_odds": 2.5, "exclude": []},
        {"label": "Away e>5% c>3.0", "min_edge": 0.05, "min_odds": 3.0, "exclude": []},
        {"label": "Away e>5% c>3.5", "min_edge": 0.05, "min_odds": 3.5, "exclude": []},
        {"label": "Away e>8% c>2.5", "min_edge": 0.08, "min_odds": 2.5, "exclude": []},
        {"label": "Away e>5% no SP1", "min_edge": 0.05, "min_odds": 2.5, "exclude": ["SP1"]},
        {"label": "Away e>5% I1+F1", "min_edge": 0.05, "min_odds": 2.5,
         "exclude": ["SP1", "E0", "D1"]},
    ]

    print(f"\n   {'Config':<22} {'Total paris':>11} {'Win%':>7} {'ROI':>8} {'Profit':>10} {'Seasons+':>9}")
    print(f"   {'-'*69}")

    for cfg in configs:
        cfg_bets = 0
        cfg_profit = 0
        cfg_seasons_pos = 0
        cfg_seasons_total = 0

        for i in range(1, len(seasons)):
            train_seasons = seasons[:i]
            test_season = seasons[i]
            train_df = df[df["season"].isin(train_seasons)]
            test_df = df[df["season"] == test_season]

            if len(train_df) < 200 or len(test_df) < 50:
                continue

            X_train = train_df[feature_cols]
            y_train = train_df["target"]
            X_test = test_df[feature_cols]

            model = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=42, verbosity=0,
            )
            model.fit(X_train, y_train)
            cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
            cal.fit(X_train, y_train)
            probs = cal.predict_proba(X_test)

            r = backtest_away(test_df, probs,
                              min_edge=cfg["min_edge"],
                              min_odds=cfg["min_odds"],
                              exclude_leagues=cfg["exclude"])
            cfg_bets += r["n"]
            cfg_profit += r["profit"]
            cfg_seasons_total += 1
            if r["n"] > 0 and r["roi"] > 0:
                cfg_seasons_pos += 1

        if cfg_bets > 0:
            cfg_roi = cfg_profit / (cfg_bets * 10) * 100
            cfg_wr = ""  # Hard to aggregate
            marker = " $$$" if cfg_roi > 0 else ""
            print(f"   {cfg['label']:<22} {cfg_bets:>11} {'':>7} {cfg_roi:>+7.1f}% "
                  f"{cfg_profit:>+9.1f} {cfg_seasons_pos}/{cfg_seasons_total}{marker}")

    print("\n" + "=" * 60)
    print("VALIDATION TERMINEE")
    print("=" * 60)
    print("\n   INTERPRETATION :")
    print("   - ROI positif sur 3+ saisons = signal REEL")
    print("   - ROI positif sur 1-2 saisons = probablement du BRUIT")
    print("   - ROI negatif partout = pas d'edge")


if __name__ == "__main__":
    walk_forward_validation()
