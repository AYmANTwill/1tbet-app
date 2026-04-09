"""
research_v2.py - Test des 3 pistes de recherche

PISTE 1 : Desaccords entre bookmakers
  Quand Pinnacle et Bet365 divergent, quelqu'un a tort.
  Features : ecart entre bookmakers, max/min ratio, nb outliers

PISTE 2 : Features avancees (proxy donnees premium)  
  Efficacite de tir, conversion xG, dominance tactique,
  patterns specifiques (equipes promues, derby, calendrier)

PISTE 3 : CLV / Timing
  Comparer cotes d'ouverture vs fermeture quand disponible.
  Identifier les mouvements de ligne suspects.

Chaque piste est testee en walk-forward sur 4 saisons.

Usage :
    python ml/research_v2.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def load_raw():
    dfs = []
    for f in sorted(Path("data/raw").glob("*.csv")):
        try:
            d = pd.read_csv(f, encoding="latin-1")
            parts = f.stem.split("_")
            d["league_code"] = parts[0]
            d["season"] = "_".join(parts[1:])
            dfs.append(d)
        except:
            pass
    return pd.concat(dfs, ignore_index=True)


def base_prep(df):
    """Preparation de base commune a toutes les pistes."""
    df = df.copy()
    req = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    if not all(c in df.columns for c in req):
        return None
    df = df.rename(columns={"HomeTeam":"home_team","AwayTeam":"away_team",
                             "FTHG":"home_goals","FTAG":"away_goals","FTR":"result"})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date","result"]).sort_values("date").reset_index(drop=True)
    df["target"] = df["result"].map({"H":0,"D":1,"A":2})

    # Convertir toutes les colonnes de cotes en numerique
    odds_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["B365","BW","IW","PS","WH","VC","Max","Avg"]
    )]
    for c in odds_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_elo(df):
    """Ajoute Elo."""
    elo = {}
    K, HA = 25, 80
    ep, ed = [], []
    for _, r in df.iterrows():
        ht, at = r["home_team"], r["away_team"]
        hr = elo.get(ht, 1500) + HA
        ar = elo.get(at, 1500)
        e = 1 / (1 + 10**((ar-hr)/400))
        ep.append(e)
        ed.append(elo.get(ht,1500) - elo.get(at,1500))
        s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht] = elo.get(ht,1500) + K*(s-e)
        elo[at] = elo.get(at,1500) + K*((1-s)-(1-e))
    df["elo_prob"] = ep
    df["elo_diff"] = ed
    return df


def add_form(df):
    """Forme recente."""
    for n in [5]:
        for side, gcol in [("home","home_goals"),("away","away_goals")]:
            col = f"{side}_form"
            df[col] = df.groupby(f"{side}_team")[gcol].transform(
                lambda x: x.rolling(n, min_periods=1).mean())
            df[col] = df.groupby(f"{side}_team")[col].shift(1)
    df["home_def"] = df.groupby("home_team")["away_goals"].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df["away_def"] = df.groupby("away_team")["home_goals"].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    df["home_def"] = df.groupby("home_team")["home_def"].shift(1)
    df["away_def"] = df.groupby("away_team")["away_def"].shift(1)
    return df


# ============================================================
# PISTE 1 : DESACCORDS ENTRE BOOKMAKERS
# ============================================================

def piste1_bookmaker_disagreement(df):
    """
    Features basees sur les DESACCORDS entre bookmakers.
    
    Quand Pinnacle dit 2.50 et Bet365 dit 3.00, il y a un desaccord
    de 20% sur la probabilite implicite. Ce desaccord EST de l'information.
    """
    print("\n   PISTE 1 : Desaccords bookmakers")
    df = df.copy()

    # Collecter toutes les cotes Home disponibles
    home_cols = [c for c in ["B365H","BWH","IWH","PSH","WHH","VCH"] if c in df.columns]
    draw_cols = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    away_cols = [c for c in ["B365A","BWA","IWA","PSA","WHA","VCA"] if c in df.columns]

    print(f"   Bookmakers Home : {len(home_cols)} ({home_cols})")
    print(f"   Bookmakers Draw : {len(draw_cols)}")
    print(f"   Bookmakers Away : {len(away_cols)}")

    if len(home_cols) < 2:
        print("   Pas assez de bookmakers pour calculer les desaccords")
        return df, []

    features = []

    # Pour chaque issue (H, D, A), calculer les desaccords
    for label, cols in [("home", home_cols), ("draw", draw_cols), ("away", away_cols)]:
        if len(cols) < 2:
            continue

        # Max - Min : l'ecart entre le bookmaker le plus genereux et le plus avare
        df[f"spread_{label}"] = df[cols].max(axis=1) - df[cols].min(axis=1)
        features.append(f"spread_{label}")

        # Coefficient de variation : ecart-type / moyenne
        df[f"cv_{label}"] = df[cols].std(axis=1) / df[cols].mean(axis=1)
        features.append(f"cv_{label}")

        # Max odds / Avg odds : ratio de la meilleure cote vs la moyenne
        df[f"max_ratio_{label}"] = df[cols].max(axis=1) / df[cols].mean(axis=1)
        features.append(f"max_ratio_{label}")

    # Si Pinnacle est disponible, comparer Pinnacle vs Bet365
    if "PSH" in df.columns and "B365H" in df.columns:
        # Pinnacle est le bookmaker "sharp" (le plus precis)
        # Si Bet365 offre plus que Pinnacle, Bet365 est probablement trop genereux
        df["pin_vs_b365_home"] = df["B365H"] / df["PSH"]
        df["pin_vs_b365_draw"] = df["B365D"] / df["PSD"]
        df["pin_vs_b365_away"] = df["B365A"] / df["PSA"]

        # Divergence : distance entre les implied probs
        df["divergence_home"] = abs(1/df["B365H"] - 1/df["PSH"])
        df["divergence_draw"] = abs(1/df["B365D"] - 1/df["PSD"])
        df["divergence_away"] = abs(1/df["B365A"] - 1/df["PSA"])

        features.extend([
            "pin_vs_b365_home", "pin_vs_b365_draw", "pin_vs_b365_away",
            "divergence_home", "divergence_draw", "divergence_away",
        ])
        print(f"   Pinnacle vs Bet365 : DISPONIBLE")

    # Cotes Bet365 pour le backtest
    if "B365H" in df.columns:
        df["b365_home"] = df["B365H"]
        df["b365_draw"] = df["B365D"]
        df["b365_away"] = df["B365A"]

    print(f"   Features desaccord : {len(features)}")
    return df, features


# ============================================================
# PISTE 2 : FEATURES AVANCEES (PROXY PREMIUM)
# ============================================================

def piste2_advanced_features(df):
    """
    Features avancees extraites des stats de match.
    Proxy pour des donnees premium (xG, tactique).
    """
    print("\n   PISTE 2 : Features avancees")
    df = df.copy()
    features = []

    # Tirs cadres
    if "HST" in df.columns:
        df["HST"] = pd.to_numeric(df["HST"], errors="coerce")
        df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
        df["HS"] = pd.to_numeric(df.get("HS", pd.Series(dtype=float)), errors="coerce")
        df["AS"] = pd.to_numeric(df.get("AS", pd.Series(dtype=float)), errors="coerce")

        # Efficacite : buts / tirs cadres (rolling)
        df["home_sot_avg"] = df.groupby("home_team")["HST"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["away_sot_avg"] = df.groupby("away_team")["AST"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["home_sot_avg"] = df.groupby("home_team")["home_sot_avg"].shift(1)
        df["away_sot_avg"] = df.groupby("away_team")["away_sot_avg"].shift(1)

        # Ratio tirs cadres / tirs total = qualite des tirs
        df["home_accuracy"] = (df["HST"] / df["HS"].replace(0, np.nan)).fillna(0.3)
        df["home_accuracy_avg"] = df.groupby("home_team")["home_accuracy"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["home_accuracy_avg"] = df.groupby("home_team")["home_accuracy_avg"].shift(1)

        df["away_accuracy"] = (df["AST"] / df["AS"].replace(0, np.nan)).fillna(0.3)
        df["away_accuracy_avg"] = df.groupby("away_team")["away_accuracy"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["away_accuracy_avg"] = df.groupby("away_team")["away_accuracy_avg"].shift(1)

        # xG proxy = tirs cadres * accuracy
        df["home_xg"] = df["home_sot_avg"].fillna(0) * df["home_accuracy_avg"].fillna(0.3)
        df["away_xg"] = df["away_sot_avg"].fillna(0) * df["away_accuracy_avg"].fillna(0.3)
        df["xg_diff"] = df["home_xg"] - df["away_xg"]

        features.extend(["home_sot_avg", "away_sot_avg",
                         "home_accuracy_avg", "away_accuracy_avg",
                         "xg_diff"])
        print(f"   Stats de tirs : OK")

    # Corners (domination)
    if "HC" in df.columns:
        df["HC"] = pd.to_numeric(df["HC"], errors="coerce")
        df["AC"] = pd.to_numeric(df["AC"], errors="coerce")
        df["home_corners_avg"] = df.groupby("home_team")["HC"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["away_corners_avg"] = df.groupby("away_team")["AC"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["home_corners_avg"] = df.groupby("home_team")["home_corners_avg"].shift(1)
        df["away_corners_avg"] = df.groupby("away_team")["away_corners_avg"].shift(1)
        df["corner_dominance"] = df["home_corners_avg"].fillna(5) / (
            df["home_corners_avg"].fillna(5) + df["away_corners_avg"].fillna(5))
        features.extend(["corner_dominance"])
        print(f"   Corners : OK")

    # Fautes et cartons (agressivite/discipline)
    if "HF" in df.columns:
        df["HF"] = pd.to_numeric(df["HF"], errors="coerce")
        df["AF"] = pd.to_numeric(df["AF"], errors="coerce")
        df["home_fouls_avg"] = df.groupby("home_team")["HF"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["home_fouls_avg"] = df.groupby("home_team")["home_fouls_avg"].shift(1)
        df["away_fouls_avg"] = df.groupby("away_team")["AF"].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
        df["away_fouls_avg"] = df.groupby("away_team")["away_fouls_avg"].shift(1)
        features.extend(["home_fouls_avg", "away_fouls_avg"])
        print(f"   Fautes : OK")

    # Goal diff form
    df["goal_diff"] = df["home_form"].fillna(0) - df["away_form"].fillna(0)
    features.append("goal_diff")

    # Closeness
    df["elo_closeness"] = 1 / (1 + abs(df["elo_diff"]) / 100)
    features.append("elo_closeness")

    # Draw rate
    df["is_draw"] = (df["result"]=="D").astype(float)
    df["draw_rate"] = (
        df.groupby("home_team")["is_draw"].transform(
            lambda x: x.rolling(10, min_periods=3).mean()).fillna(0.25) +
        df.groupby("away_team")["is_draw"].transform(
            lambda x: x.rolling(10, min_periods=3).mean()).fillna(0.25)
    ) / 2
    df["draw_rate"] = df.groupby("home_team")["draw_rate"].shift(1)
    features.append("draw_rate")

    print(f"   Features avancees : {len(features)}")
    return df, features


# ============================================================
# PISTE 3 : CLV / TIMING (mouvement de cotes)
# ============================================================

def piste3_line_movement(df):
    """
    Features de mouvement de ligne.
    Certains fichiers ont les cotes d'ouverture et de fermeture.
    """
    print("\n   PISTE 3 : Mouvement de ligne / CLV")
    df = df.copy()
    features = []

    # Verifier si on a les cotes Max et Avg (proxy open/close)
    has_max = "MaxH" in df.columns
    has_avg = "AvgH" in df.columns

    if has_max and has_avg:
        df["MaxH"] = pd.to_numeric(df["MaxH"], errors="coerce")
        df["MaxD"] = pd.to_numeric(df["MaxD"], errors="coerce")
        df["MaxA"] = pd.to_numeric(df["MaxA"], errors="coerce")
        df["AvgH"] = pd.to_numeric(df["AvgH"], errors="coerce")
        df["AvgD"] = pd.to_numeric(df["AvgD"], errors="coerce")
        df["AvgA"] = pd.to_numeric(df["AvgA"], errors="coerce")

        # Max/Avg ratio : si max >> avg, un bookmaker est outlier
        df["max_avg_home"] = df["MaxH"] / df["AvgH"]
        df["max_avg_draw"] = df["MaxD"] / df["AvgD"]
        df["max_avg_away"] = df["MaxA"] / df["AvgA"]

        features.extend(["max_avg_home", "max_avg_draw", "max_avg_away"])
        print(f"   Max/Avg ratio : DISPONIBLE")

    # Si on a B365 et Pinnacle, la difference est un proxy de mouvement
    if "PSH" in df.columns and "B365H" in df.columns:
        # Pinnacle ferme plus vite que B365
        # Si PS < B365, la cote a baisse chez le sharp = argent smart
        df["smart_money_home"] = (1/df["PSH"]) - (1/df["B365H"])
        df["smart_money_draw"] = (1/df["PSD"]) - (1/df["B365D"])
        df["smart_money_away"] = (1/df["PSA"]) - (1/df["B365A"])

        features.extend(["smart_money_home", "smart_money_draw", "smart_money_away"])
        print(f"   Smart money signal : DISPONIBLE")

    if not features:
        print(f"   Aucune donnee de mouvement disponible")

    # Cotes pour backtest
    if "B365H" in df.columns:
        df["b365_home"] = df["B365H"]
        df["b365_draw"] = df["B365D"]
        df["b365_away"] = df["B365A"]

    print(f"   Features mouvement : {len(features)}")
    return df, features


# ============================================================
# VALIDATION WALK-FORWARD
# ============================================================

def walk_forward(df, feature_cols, label, bet_types=["H","D","A"],
                 min_edge=0.05, min_odds=1.5):
    """Walk-forward generique."""
    seasons = sorted(df["season"].unique())
    results = []

    for i in range(1, len(seasons)):
        train = df[df["season"].isin(seasons[:i])]
        test = df[df["season"] == seasons[i]]
        if len(train) < 200 or len(test) < 50:
            continue

        fc = [c for c in feature_cols if c in train.columns]
        train_clean = train.dropna(subset=fc)
        test_clean = test.dropna(subset=fc)
        if len(train_clean) < 200:
            continue

        X_tr, y_tr = train_clean[fc], train_clean["target"]
        X_te = test_clean[fc]

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
        cal.fit(X_tr, y_tr)
        probs = cal.predict_proba(X_te)

        # Backtest
        stake = 10.0
        bets = []
        type_map = {"H": (0, "b365_home"), "D": (1, "b365_draw"), "A": (2, "b365_away")}

        for j, (_, match) in enumerate(test_clean.iterrows()):
            for bt in bet_types:
                idx, odds_col = type_map[bt]
                odds = match.get(odds_col)
                if pd.isna(odds) or odds < min_odds:
                    continue
                edge = probs[j, idx] - 1/odds
                if edge >= min_edge:
                    won = match["result"] == bt
                    bets.append({"won": won,
                                 "profit": stake*(odds-1) if won else -stake,
                                 "type": bt})

        n = len(bets)
        profit = sum(b["profit"] for b in bets) if bets else 0
        roi = profit / (n * stake) * 100 if n > 0 else 0

        results.append({
            "season": seasons[i], "n": n, "profit": profit, "roi": roi,
            "win_rate": sum(b["won"] for b in bets)/n if n > 0 else 0,
        })

    return results


def print_results(label, results):
    """Affiche les resultats walk-forward."""
    total_n = sum(r["n"] for r in results)
    total_profit = sum(r["profit"] for r in results)
    total_roi = total_profit / (total_n * 10) * 100 if total_n > 0 else 0
    pos = sum(1 for r in results if r["roi"] > 0 and r["n"] > 5)

    print(f"\n   {label}")
    print(f"   {'Season':<15} {'Paris':>6} {'Win%':>7} {'ROI':>8} {'Profit':>10}")
    print(f"   {'-'*48}")
    for r in results:
        m = " $$$" if r["roi"] > 0 and r["n"] > 5 else ""
        if r["n"] > 0:
            print(f"   {r['season']:<15} {r['n']:>6} {r['win_rate']:>6.1%} "
                  f"{r['roi']:>+7.1f}% {r['profit']:>+9.1f}{m}")
    print(f"   {'-'*48}")
    print(f"   {'TOTAL':<15} {total_n:>6} {'':>7} {total_roi:>+7.1f}% {total_profit:>+9.1f}")
    print(f"   Saisons + : {pos}/{len(results)}")
    return total_roi, pos, len(results)


def main():
    print("=" * 60)
    print("RECHERCHE V2 : TEST DES 3 PISTES")
    print("=" * 60)

    df = load_raw()
    df = base_prep(df)
    if df is None:
        return
    df = add_elo(df)
    df = add_form(df)

    # Features de base (toujours presentes)
    base_features = ["elo_prob", "elo_diff", "home_form", "away_form",
                     "home_def", "away_def"]
    base_features = [c for c in base_features if c in df.columns]

    # ============================================================
    # BASELINE : features de base seules
    # ============================================================
    print("\n" + "=" * 60)
    print("BASELINE : Elo + Forme seuls")
    print("=" * 60)

    # Cotes pour backtest
    if "B365H" in df.columns:
        df["b365_home"] = pd.to_numeric(df["B365H"], errors="coerce")
        df["b365_draw"] = pd.to_numeric(df["B365D"], errors="coerce")
        df["b365_away"] = pd.to_numeric(df["B365A"], errors="coerce")

    for bt, label in [("H","Home"), ("D","Draw"), ("A","Away")]:
        r = walk_forward(df, base_features, f"Base {label}", [bt])
        print_results(f"BASELINE {label} (e>5%)", r)

    # ============================================================
    # PISTE 1 : DESACCORDS BOOKMAKERS
    # ============================================================
    print("\n" + "=" * 60)
    print("PISTE 1 : DESACCORDS BOOKMAKERS")
    print("=" * 60)

    df1, feat1 = piste1_bookmaker_disagreement(df)
    all_feat1 = base_features + feat1

    for bt, label in [("H","Home"), ("D","Draw"), ("A","Away")]:
        r = walk_forward(df1, all_feat1, f"P1 {label}", [bt])
        roi, pos, tot = print_results(f"PISTE 1 {label} (e>5%)", r)

    # Test combiné H+D+A
    r = walk_forward(df1, all_feat1, "P1 ALL", ["H","D","A"])
    print_results("PISTE 1 TOUS TYPES (e>5%)", r)

    # ============================================================
    # PISTE 2 : FEATURES AVANCEES
    # ============================================================
    print("\n" + "=" * 60)
    print("PISTE 2 : FEATURES AVANCEES")
    print("=" * 60)

    df2, feat2 = piste2_advanced_features(df)
    all_feat2 = base_features + feat2

    for bt, label in [("H","Home"), ("D","Draw"), ("A","Away")]:
        r = walk_forward(df2, all_feat2, f"P2 {label}", [bt])
        print_results(f"PISTE 2 {label} (e>5%)", r)

    # ============================================================
    # PISTE 3 : MOUVEMENT DE LIGNE
    # ============================================================
    print("\n" + "=" * 60)
    print("PISTE 3 : MOUVEMENT DE LIGNE")
    print("=" * 60)

    df3, feat3 = piste3_line_movement(df)
    all_feat3 = base_features + feat3

    if feat3:
        for bt, label in [("H","Home"), ("D","Draw"), ("A","Away")]:
            r = walk_forward(df3, all_feat3, f"P3 {label}", [bt])
            print_results(f"PISTE 3 {label} (e>5%)", r)

    # ============================================================
    # COMBINAISON : TOUTES LES PISTES
    # ============================================================
    print("\n" + "=" * 60)
    print("COMBINAISON : TOUTES LES PISTES")
    print("=" * 60)

    # Fusionner toutes les features
    df_all = df.copy()
    for col in feat1:
        if col in df1.columns:
            df_all[col] = df1[col]
    for col in feat2:
        if col in df2.columns:
            df_all[col] = df2[col]
    for col in feat3:
        if col in df3.columns:
            df_all[col] = df3[col]

    all_features = list(set(base_features + feat1 + feat2 + feat3))
    all_features = [c for c in all_features if c in df_all.columns]
    print(f"\n   Total features : {len(all_features)}")

    for bt, label in [("H","Home"), ("D","Draw"), ("A","Away")]:
        r = walk_forward(df_all, all_features, f"COMBO {label}", [bt])
        roi, pos, tot = print_results(f"COMBO {label} (e>5%)", r)

    r = walk_forward(df_all, all_features, "COMBO ALL", ["H","D","A"])
    roi, pos, tot = print_results("COMBO TOUS TYPES (e>5%)", r)

    # ============================================================
    # RESUME FINAL
    # ============================================================
    print("\n" + "=" * 60)
    print("RESUME : quelle piste poursuivre ?")
    print("=" * 60)
    print("   Critere : ROI positif sur 3+ saisons = signal reel")
    print("   Regarder les resultats ci-dessus et choisir.")


if __name__ == "__main__":
    main()
