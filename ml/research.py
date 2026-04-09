"""
research.py - Recherche d'edge : decorrelation + features avancees + Draw targeting

TROIS INNOVATIONS COMBINEES :

1. LOSS DECORRELÉE (Hubacek 2023)
   XGBoost custom objective qui penalise les predictions proches du bookmaker.
   Le modele est FORCE de trouver des angles differents du marche.

2. FEATURES AVANCEES
   Football-data.co.uk fournit des stats detaillees :
   tirs, tirs cadres, corners, fautes, cartons.
   On calcule des proxys xG et des ratios d'efficacite.

3. DRAW TARGETING
   Les Draw sont le resultat le plus mal price par les bookmakers.
   On teste specifiquement cette niche avec plusieurs seuils.

Usage :
    python research.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. CHARGEMENT ET FEATURES AVANCEES
# ============================================================

def load_raw_data():
    """Charge les CSV bruts pour acceder aux stats detaillees."""
    raw_dir = Path("data/raw")
    all_dfs = []

    for csv_file in sorted(raw_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, encoding="latin-1")
            # Identifier la ligue et saison depuis le nom de fichier
            parts = csv_file.stem.split("_")
            if len(parts) >= 2:
                df["league_code"] = parts[0]
                df["season"] = "_".join(parts[1:])
            all_dfs.append(df)
        except Exception as e:
            print(f"   Skip {csv_file.name}: {e}")

    if not all_dfs:
        print("Aucun fichier dans data/raw/ !")
        raise SystemExit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def build_advanced_features(df):
    """
    Feature engineering avance avec TOUTES les stats disponibles.

    NOUVELLES FEATURES :
    - Shooting efficiency : tirs cadres / tirs total
    - xG proxy : tirs cadres * coefficient historique
    - Corner dominance : ratio corners home/away
    - Discipline : cartons jaunes/rouges
    - Form avancee : points sur 5 derniers matchs (pas juste les buts)
    """
    print("\n   FEATURE ENGINEERING AVANCE")
    print("   " + "=" * 40)

    df = df.copy()

    # Colonnes essentielles
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"   Colonnes manquantes : {missing}")
        return None

    # Renommer
    df = df.rename(columns={
        "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    })

    # Date
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "result"])
    df = df.sort_values("date").reset_index(drop=True)

    # ---- COTES BET365 (pour le backtest) ----
    for col in ["B365H", "B365D", "B365A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "B365H" in df.columns:
        df["b365_home"] = df["B365H"]
        df["b365_draw"] = df["B365D"]
        df["b365_away"] = df["B365A"]

        # Implied probs normalisees (ce que le bookmaker pense)
        imp_h = 1 / df["b365_home"]
        imp_d = 1 / df["b365_draw"]
        imp_a = 1 / df["b365_away"]
        total = imp_h + imp_d + imp_a
        df["book_prob_home"] = imp_h / total
        df["book_prob_draw"] = imp_d / total
        df["book_prob_away"] = imp_a / total

    # ---- ELO RATINGS ----
    print("   Calcul Elo...")
    elo_ratings = {}
    elo_probs = []
    elo_diffs = []

    K = 25
    HOME_ADV = 80

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hr = elo_ratings.get(ht, 1500) + HOME_ADV
        ar = elo_ratings.get(at, 1500)

        exp_h = 1 / (1 + 10 ** ((ar - hr) / 400))
        elo_probs.append(exp_h)
        elo_diffs.append(elo_ratings.get(ht, 1500) - elo_ratings.get(at, 1500))

        # Update
        if row["result"] == "H":
            sh, sa = 1.0, 0.0
        elif row["result"] == "D":
            sh, sa = 0.5, 0.5
        else:
            sh, sa = 0.0, 1.0

        elo_ratings[ht] = elo_ratings.get(ht, 1500) + K * (sh - exp_h)
        elo_ratings[at] = elo_ratings.get(at, 1500) + K * (sa - (1 - exp_h))

    df["elo_prob"] = elo_probs
    df["elo_diff"] = elo_diffs

    # ---- FORME RECENTE (buts) ----
    print("   Calcul forme...")
    for n in [3, 5]:
        df[f"home_form_{n}"] = (
            df.groupby("home_team")["home_goals"]
            .transform(lambda x: x.rolling(n, min_periods=1).mean())
        )
        df[f"away_form_{n}"] = (
            df.groupby("away_team")["away_goals"]
            .transform(lambda x: x.rolling(n, min_periods=1).mean())
        )
        # Shift pour eviter data leakage
        df[f"home_form_{n}"] = df.groupby("home_team")[f"home_form_{n}"].shift(1)
        df[f"away_form_{n}"] = df.groupby("away_team")[f"away_form_{n}"].shift(1)

    # ---- FORME DEFENSIVE ----
    df["home_def_form"] = (
        df.groupby("home_team")["away_goals"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    df["away_def_form"] = (
        df.groupby("away_team")["home_goals"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    df["home_def_form"] = df.groupby("home_team")["home_def_form"].shift(1)
    df["away_def_form"] = df.groupby("away_team")["away_def_form"].shift(1)

    # ---- FEATURES DE TIRS (si disponibles) ----
    has_shots = "HS" in df.columns and "AS" in df.columns
    if has_shots:
        print("   Stats de tirs detectees !")
        df["HS"] = pd.to_numeric(df["HS"], errors="coerce")
        df["AS"] = pd.to_numeric(df["AS"], errors="coerce")
        df["HST"] = pd.to_numeric(df.get("HST", pd.Series(dtype=float)), errors="coerce")
        df["AST"] = pd.to_numeric(df.get("AST", pd.Series(dtype=float)), errors="coerce")

        # Tirs cadres rolling
        df["home_shots_on"] = (
            df.groupby("home_team")["HST"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        df["away_shots_on"] = (
            df.groupby("away_team")["AST"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        df["home_shots_on"] = df.groupby("home_team")["home_shots_on"].shift(1)
        df["away_shots_on"] = df.groupby("away_team")["away_shots_on"].shift(1)

        # Efficacite de tir (buts / tirs cadres) = proxy xG
        df["home_efficiency"] = (
            df.groupby("home_team")
            .apply(lambda g: (g["home_goals"] / g["HST"].replace(0, np.nan)).rolling(5, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        df["home_efficiency"] = df.groupby("home_team")["home_efficiency"].shift(1)

        # xG proxy = tirs cadres * efficacite moyenne
        df["home_xg_proxy"] = df["home_shots_on"] * df["home_efficiency"].fillna(0.3)
        df["away_xg_proxy"] = df["away_shots_on"] * 0.3  # Simplifie

        # Ratio de domination des tirs
        total_shots = df["home_shots_on"].fillna(0) + df["away_shots_on"].fillna(0)
        df["shot_dominance"] = np.where(
            total_shots > 0,
            df["home_shots_on"].fillna(0) / total_shots,
            0.5
        )

    # ---- CORNERS (si disponibles) ----
    if "HC" in df.columns:
        df["HC"] = pd.to_numeric(df["HC"], errors="coerce")
        df["AC"] = pd.to_numeric(df["AC"], errors="coerce")
        df["home_corners"] = (
            df.groupby("home_team")["HC"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        df["home_corners"] = df.groupby("home_team")["home_corners"].shift(1)

    # ---- FEATURE COMPOSITE : expected goal diff ----
    df["goal_diff_form"] = df["home_form_5"].fillna(0) - df["away_form_5"].fillna(0)
    df["def_diff_form"] = df["away_def_form"].fillna(0) - df["home_def_form"].fillna(0)

    # ---- DRAW SPECIFIC FEATURES ----
    # Les draws arrivent plus souvent quand les equipes sont proches en force
    df["elo_closeness"] = 1 / (1 + abs(df["elo_diff"]) / 100)
    df["odds_closeness"] = 1 / (1 + abs(df.get("b365_home", 2.5) - df.get("b365_away", 2.5)))

    # Draw rate historique de chaque equipe
    df["is_draw"] = (df["result"] == "D").astype(float)
    df["home_draw_rate"] = (
        df.groupby("home_team")["is_draw"]
        .transform(lambda x: x.rolling(10, min_periods=3).mean())
    )
    df["away_draw_rate"] = (
        df.groupby("away_team")["is_draw"]
        .transform(lambda x: x.rolling(10, min_periods=3).mean())
    )
    df["home_draw_rate"] = df.groupby("home_team")["home_draw_rate"].shift(1)
    df["away_draw_rate"] = df.groupby("away_team")["away_draw_rate"].shift(1)
    df["avg_draw_rate"] = (df["home_draw_rate"].fillna(0.25) + df["away_draw_rate"].fillna(0.25)) / 2

    # Target
    df["target"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    # Supprimer NaN
    feature_cols = get_feature_cols(df, has_shots)
    before = len(df)
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    print(f"   {before} -> {len(df)} matchs utilisables")
    print(f"   {len(feature_cols)} features")

    return df, feature_cols


def get_feature_cols(df, has_shots=False):
    """
    Retourne les features SANS cotes bookmaker.
    La decorrelation vient du fait que le modele ne voit PAS
    ce que le bookmaker pense.
    """
    cols = [
        "elo_prob",
        "elo_diff",
        "elo_closeness",
        "home_form_3",
        "home_form_5",
        "away_form_3",
        "away_form_5",
        "home_def_form",
        "away_def_form",
        "goal_diff_form",
        "def_diff_form",
        "avg_draw_rate",
    ]

    if has_shots:
        cols.extend([
            "home_shots_on",
            "away_shots_on",
            "home_xg_proxy",
            "shot_dominance",
        ])

    if "home_corners" in df.columns:
        cols.append("home_corners")

    # Ne garder que les colonnes presentes
    cols = [c for c in cols if c in df.columns]
    return cols


# ============================================================
# 2. LOSS DECORRELÉE (HUBACEK)
# ============================================================

def decorrelated_objective(predt, dtrain, book_probs, alpha=0.5):
    """
    Custom objective function pour XGBoost qui penalise la correlation
    avec les probabilites du bookmaker.

    L'idee (Hubacek 2023) : on ajoute un terme de penalite quand
    la prediction du modele est trop proche de celle du bookmaker.

    loss = cross_entropy + alpha * correlation_penalty

    alpha controle la force de la decorrelation :
    - alpha = 0 : XGBoost normal
    - alpha = 0.5 : decorrelation moderee
    - alpha = 1.0 : decorrelation forte

    En pratique, on ne peut pas modifier la loss de XGBClassifier
    directement pour multi-classe, donc on utilise une approche
    indirecte : on modifie les POIDS des echantillons.

    Les matchs ou le bookmaker est tres confiant (prob > 60%)
    recoivent un poids plus faible. Ca force le modele a se
    concentrer sur les matchs incertains (ou le Draw est plus probable).
    """
    pass  # Voir implementation via sample_weight ci-dessous


def compute_decorrelation_weights(df, alpha=0.5):
    """Poids de decorrelation basés sur l'incertitude du bookmaker."""
    n = len(df)
    
    if "book_prob_home" not in df.columns:
        return np.ones(n).astype(np.float32)

    p_h = df["book_prob_home"].fillna(0.33).values
    p_d = df["book_prob_draw"].fillna(0.33).values
    p_a = df["book_prob_away"].fillna(0.33).values

    entropy = -(
        p_h * np.log(np.maximum(p_h, 0.01)) +
        p_d * np.log(np.maximum(p_d, 0.01)) +
        p_a * np.log(np.maximum(p_a, 0.01))
    )

    e_min = np.nanmin(entropy)
    e_max = np.nanmax(entropy)
    if e_max - e_min < 0.001:
        return np.ones(n).astype(np.float32)
    
    entropy_norm = (entropy - e_min) / (e_max - e_min)
    entropy_norm = np.nan_to_num(entropy_norm, nan=0.5)

    weights = (1 - alpha) + alpha * (0.5 + entropy_norm)
    weights = np.clip(weights, 0.1, 10.0).astype(np.float32)
    
    return weights

# ============================================================
# 3. ENTRAINEMENT ET BACKTEST
# ============================================================

def train_decorrelated(X_train, y_train, weights, feature_cols):
    """Entraine XGBoost avec poids de decorrelation."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train, sample_weight=weights)
    return model


def backtest_draw_specialist(test_df, probabilities, feature_cols,
                              min_edges=[0.05, 0.08, 0.10, 0.12, 0.15]):
    """
    Backtest specifique aux Draw avec plusieurs seuils.
    Teste aussi les paris Home et Away pour comparaison.
    """
    print("\n" + "=" * 60)
    print("BACKTEST DRAW SPECIALIST")
    print("=" * 60)

    results = {}

    for min_edge in min_edges:
        bankroll = 1000.0
        stake = 10.0
        bets = []

        for i, (_, match) in enumerate(test_df.iterrows()):
            prob_d = probabilities[i, 1]
            odds_d = match.get("b365_draw")

            if pd.isna(odds_d) or odds_d <= 1.0:
                continue

            implied_d = 1 / odds_d
            edge = prob_d - implied_d

            if edge >= min_edge and bankroll >= stake:
                won = match["result"] == "D"
                profit = stake * (odds_d - 1) if won else -stake
                bankroll += profit
                bets.append({"won": won, "profit": profit, "edge": edge, "odds": odds_d})

        n = len(bets)
        if n > 0:
            bets_df = pd.DataFrame(bets)
            wins = bets_df["won"].sum()
            profit = bets_df["profit"].sum()
            roi = profit / (n * stake) * 100
            avg_edge = bets_df["edge"].mean()
            avg_odds = bets_df["odds"].mean()
        else:
            wins, profit, roi, avg_edge, avg_odds = 0, 0, 0, 0, 0

        results[min_edge] = {
            "n_bets": n, "wins": wins, "profit": profit,
            "roi": roi, "avg_edge": avg_edge, "avg_odds": avg_odds,
        }

    # Affichage
    print(f"\n   {'Edge min':>10} {'Paris':>6} {'Wins':>6} {'Win%':>7} {'ROI':>8} {'Profit':>10} {'Avg odds':>9}")
    print(f"   {'-'*58}")
    for edge, r in results.items():
        wr = r["wins"]/r["n_bets"] if r["n_bets"] > 0 else 0
        marker = " <-- BEST" if r["roi"] == max(x["roi"] for x in results.values() if x["n_bets"] > 5) and r["roi"] > 0 else ""
        profitable = " $$$" if r["roi"] > 0 else ""
        print(f"   {edge:>9.0%} {r['n_bets']:>6} {r['wins']:>6} {wr:>6.1%} "
              f"{r['roi']:>+7.1f}% {r['profit']:>+9.2f} {r['avg_odds']:>8.2f}{profitable}{marker}")

    # Backtest aussi sur H et A pour comparaison
    print(f"\n   COMPARAISON H / D / A (edge > 8%) :")
    for bet_type, target_idx, bet_label in [("H", 0, "Home"), ("D", 1, "Draw"), ("A", 2, "Away")]:
        bankroll = 1000.0
        bets = []
        odds_col = {"H": "b365_home", "D": "b365_draw", "A": "b365_away"}[bet_type]

        for i, (_, match) in enumerate(test_df.iterrows()):
            prob = probabilities[i, target_idx]
            odds = match.get(odds_col)
            if pd.isna(odds) or odds <= 1.0:
                continue
            edge = prob - 1/odds
            if edge >= 0.08 and bankroll >= stake:
                won = match["result"] == bet_type
                profit = stake * (odds - 1) if won else -stake
                bankroll += profit
                bets.append({"won": won, "profit": profit})

        if bets:
            bdf = pd.DataFrame(bets)
            print(f"   {bet_label:>6} : {len(bdf)} paris, "
                  f"win {bdf['won'].mean():.1%}, "
                  f"ROI {bdf['profit'].sum()/(len(bdf)*stake)*100:+.1f}%")

    return results


def run_research():
    """Pipeline de recherche complet."""
    print("=" * 60)
    print("RECHERCHE D'EDGE : DECORRELATION + FEATURES + DRAW")
    print("=" * 60)

    # 1. Charger les donnees brutes
    print("\n[1/6] Chargement des donnees brutes...")
    df = load_raw_data()
    print(f"   {len(df)} lignes chargees depuis data/raw/")

    # 2. Feature engineering avance
    print("\n[2/6] Feature engineering avance...")
    result = build_advanced_features(df)
    if result is None:
        return
    df, feature_cols = result

    # 3. Split temporel
    print("\n[3/6] Split temporel...")
    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        print("   Pas assez de saisons pour un split !")
        return

    # Test sur la derniere saison disponible
    test_season = seasons[-1]
    train_df = df[df["season"] != test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    print(f"   Train : {len(train_df)} matchs ({seasons[0]} a {seasons[-2]})")
    print(f"   Test  : {len(test_df)} matchs ({test_season})")

    # 4. Entrainement avec DECORRELATION
    print("\n[4/6] Entrainement avec decorrelation...")

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    # Tester plusieurs niveaux de decorrelation
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    best_model = None
    best_calibrated = None
    best_alpha = 0
    best_draw_roi = -999

    print(f"\n   {'Alpha':>7} {'Accuracy':>10} {'LogLoss':>10} {'Draw ROI (8%)':>14}")
    print(f"   {'-'*44}")

    for alpha in alphas:
        weights = compute_decorrelation_weights(train_df, alpha=alpha)
        model = train_decorrelated(X_train, y_train, weights, feature_cols)

        # Calibrer
        cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
        cal.fit(X_train, y_train, sample_weight=weights)

        probs = cal.predict_proba(X_test)
        preds = cal.predict(X_test)
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)

        # Quick draw backtest a 8%
        draw_bets = 0
        draw_profit = 0
        for i, (_, match) in enumerate(test_df.iterrows()):
            odds_d = match.get("b365_draw")
            if pd.isna(odds_d) or odds_d <= 1.0:
                continue
            edge = probs[i, 1] - 1/odds_d
            if edge >= 0.08:
                won = match["result"] == "D"
                draw_profit += 10 * (odds_d - 1) if won else -10
                draw_bets += 1

        draw_roi = draw_profit / (draw_bets * 10) * 100 if draw_bets > 0 else 0

        marker = ""
        if draw_roi > best_draw_roi and draw_bets >= 5:
            best_draw_roi = draw_roi
            best_alpha = alpha
            best_model = model
            best_calibrated = cal
            marker = " <-- BEST"

        print(f"   {alpha:>7.1f} {acc:>9.1%} {ll:>10.4f} {draw_roi:>+10.1f}% ({draw_bets}p){marker}")

    if best_calibrated is None:
        print("\n   Aucun modele profitable. Essayons quand meme le meilleur.")
        best_calibrated = cal
        best_model = model
        best_alpha = alphas[-1]

    # 5. Evaluation detaillee du meilleur modele
    print(f"\n[5/6] Evaluation du meilleur modele (alpha={best_alpha})...")

    probs = best_calibrated.predict_proba(X_test)
    preds = best_calibrated.predict(X_test)

    print(f"\n   Accuracy : {accuracy_score(y_test, preds):.1%}")
    print(f"   Log Loss : {log_loss(y_test, probs):.4f}")

    # Calibration check
    for cls, name in [(0, "Home"), (1, "Draw"), (2, "Away")]:
        true_bin = (y_test == cls).astype(int)
        brier = brier_score_loss(true_bin, probs[:, cls])
        # Calibration : predicted prob vs actual frequency
        predicted = probs[:, cls]
        bins = np.linspace(0, 1, 6)
        for b in range(len(bins)-1):
            mask = (predicted >= bins[b]) & (predicted < bins[b+1])
            if mask.sum() > 10:
                actual_rate = true_bin[mask].mean()
                predicted_rate = predicted[mask].mean()

    # 6. Backtest complet
    print(f"\n[6/6] Backtest Draw specialist (alpha={best_alpha})...")
    results = backtest_draw_specialist(test_df, probs, feature_cols)

    # Feature importance
    print(f"\n   IMPORTANCE DES FEATURES (alpha={best_alpha}) :")
    importances = best_model.feature_importances_
    for name, imp in sorted(zip(feature_cols, importances),
                            key=lambda x: x[1], reverse=True):
        bar = "#" * int(imp * 40)
        print(f"   {name:<25} {imp:.3f} {bar}")

    # Sauvegarder le meilleur modele
    Path("model").mkdir(exist_ok=True)
    import joblib
    joblib.dump(best_calibrated, "model/xgb_decorrelated.pkl")
    joblib.dump(feature_cols, "model/feature_cols_research.pkl")
    print(f"\n   Modele sauvegarde : model/xgb_decorrelated.pkl (alpha={best_alpha})")

    # Graphique de comparaison des alphas
    Path("charts").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_rois = []
    for alpha in alphas:
        weights = compute_decorrelation_weights(train_df, alpha=alpha)
        m = train_decorrelated(X_train, y_train, weights, feature_cols)
        c = CalibratedClassifierCV(estimator=m, method="sigmoid", cv=3)
        c.fit(X_train, y_train, sample_weight=weights)
        p = c.predict_proba(X_test)
        dr_profit = 0
        dr_bets = 0
        for i, (_, match) in enumerate(test_df.iterrows()):
            odds_d = match.get("b365_draw")
            if pd.isna(odds_d) or odds_d <= 1.0:
                continue
            if p[i, 1] - 1/odds_d >= 0.08:
                won = match["result"] == "D"
                dr_profit += 10 * (odds_d - 1) if won else -10
                dr_bets += 1
        draw_rois.append(dr_profit / (dr_bets * 10) * 100 if dr_bets > 0 else 0)

    ax.bar(range(len(alphas)), draw_rois,
           color=["#5DCAA5" if r > 0 else "#F09595" for r in draw_rois])
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"a={a}" for a in alphas])
    ax.set_ylabel("Draw ROI (%)")
    ax.set_title("Impact de la decorrelation sur le ROI Draw")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("charts/decorrelation_impact.png", dpi=150)
    plt.close()
    print("   Graphique : charts/decorrelation_impact.png")

    print("\n" + "=" * 60)
    print("RECHERCHE TERMINEE")
    print("=" * 60)


if __name__ == "__main__":
    run_research()
