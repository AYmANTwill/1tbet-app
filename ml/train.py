"""
train.py - Entraine XGBoost sur les vraies donnees historiques

DIFFERENCE AVEC LE REPO 3 :
- Donnees REELLES (pas simulees)
- Split temporel par saison (train = saisons anciennes, test = derniere saison)
- Evaluation plus rigoureuse avec Brier score par classe
- Sauvegarde du modele et des metriques

Usage :
    python train.py
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from features import build_features, FEATURE_COLS, PINNACLE_FEATURES


def load_data():
    """Charge les donnees fusionnees."""
    path = Path("data/matches_all.csv")
    if not path.exists():
        print("   Fichier data/matches_all.csv introuvable !")
        print("   Lance d'abord : python download_data.py")
        raise SystemExit(1)

    df = pd.read_csv(path, parse_dates=["date"])
    print(f"   Donnees chargees : {len(df)} matchs")
    return df


def temporal_split(df, test_seasons=1):
    """
    Separe par saison : les N dernieres saisons = test, le reste = train.

    C'est plus realiste qu'un split aleatoire car en production
    on entraine sur le passe et on predit le futur.
    """
    seasons = sorted(df["season"].unique())
    test_seasons_list = seasons[-test_seasons:]
    train_seasons_list = seasons[:-test_seasons]

    train = df[df["season"].isin(train_seasons_list)]
    test = df[df["season"].isin(test_seasons_list)]

    print(f"   Train : {len(train)} matchs (saisons {train_seasons_list[0]} a {train_seasons_list[-1]})")
    print(f"   Test  : {len(test)} matchs (saison {test_seasons_list[0]})")

    return train, test


def get_feature_columns(df):
    """Determine les colonnes disponibles (Pinnacle n'est pas toujours la)."""
    cols = FEATURE_COLS.copy()

    # Ajouter Pinnacle si disponible
    if all(c in df.columns for c in PINNACLE_FEATURES):
        cols.extend(PINNACLE_FEATURES)
        print(f"   Pinnacle disponible : {len(cols)} features")
    else:
        print(f"   Pinnacle indisponible : {len(cols)} features (Bet365 only)")

    return cols


def train_xgboost(X_train, y_train):
    """Entraine le modele XGBoost."""
    print("\n   Entrainement XGBoost...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train)
    return model


def calibrate(model, X_train, y_train):
    """Calibre les probabilites avec Platt Scaling."""
    print("   Calibration (Platt Scaling)...")

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=5,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate(model, calibrated, X_test, y_test, test_df):
    """Evaluation detaillee sur les donnees de test."""

    y_prob_raw = model.predict_proba(X_test)
    y_prob_cal = calibrated.predict_proba(X_test)
    y_pred_cal = calibrated.predict(X_test)

    # Metriques globales
    acc = accuracy_score(y_test, y_pred_cal)
    ll_raw = log_loss(y_test, y_prob_raw)
    ll_cal = log_loss(y_test, y_prob_cal)

    # Brier score par classe
    classes = {0: "Home", 1: "Draw", 2: "Away"}
    brier_scores = {}
    for cls_idx, cls_name in classes.items():
        true_binary = (y_test == cls_idx).astype(int)
        brier_raw = brier_score_loss(true_binary, y_prob_raw[:, cls_idx])
        brier_cal = brier_score_loss(true_binary, y_prob_cal[:, cls_idx])
        brier_scores[cls_name] = {"raw": brier_raw, "calibrated": brier_cal}

    print("\n" + "=" * 60)
    print("EVALUATION SUR DONNEES REELLES")
    print("=" * 60)
    print(f"\n   Accuracy :  {acc:.1%}")
    print(f"   Log Loss :  {ll_raw:.4f} (brut) -> {ll_cal:.4f} (calibre)")
    print(f"\n   Brier Score par classe :")
    for cls_name, scores in brier_scores.items():
        improvement = scores['raw'] - scores['calibrated']
        print(f"   {cls_name:<8} : {scores['raw']:.4f} -> {scores['calibrated']:.4f} "
              f"({'ameliore' if improvement > 0 else 'degrade'} de {abs(improvement):.4f})")

    return {
        "accuracy": acc,
        "log_loss_raw": ll_raw,
        "log_loss_calibrated": ll_cal,
        "brier_scores": brier_scores,
        "probabilities": y_prob_cal,
    }


def backtest_ev(test_df, probabilities, min_edge=0.03):
    """
    Backtest +EV sur les donnees reelles.
    Compare les probabilites du modele aux cotes Bet365.
    """
    print("\n" + "=" * 60)
    print("BACKTEST +EV SUR DONNEES REELLES")
    print("=" * 60)

    bankroll = 1000.0
    initial = bankroll
    stake = 10.0
    bets = []
    bankroll_history = [bankroll]

    for i, (_, match) in enumerate(test_df.iterrows()):
        prob_h = probabilities[i, 0]
        prob_d = probabilities[i, 1]
        prob_a = probabilities[i, 2]

        # Cotes Bet365
        odds_h = match.get("b365_home", None)
        odds_d = match.get("b365_draw", None)
        odds_a = match.get("b365_away", None)

        if pd.isna(odds_h) or pd.isna(odds_d) or pd.isna(odds_a):
            continue

        # Chercher un pari +EV
        options = [
            ("H", prob_h, odds_h, match["result"] == "H"),
            ("D", prob_d, odds_d, match["result"] == "D"),
            ("A", prob_a, odds_a, match["result"] == "A"),
        ]

        for bet_type, prob, odds, won in options:
            implied = 1 / odds
            edge = prob - implied

            if edge >= min_edge and bankroll >= stake:
                profit = stake * (odds - 1) if won else -stake
                bankroll += profit
                bankroll_history.append(bankroll)

                bets.append({
                    "match": f"{match['home_team']} vs {match['away_team']}",
                    "bet_type": bet_type,
                    "prob": round(prob, 3),
                    "odds": odds,
                    "edge": round(edge, 3),
                    "won": won,
                    "profit": round(profit, 2),
                })

    if not bets:
        print("   Aucun pari place !")
        return

    bets_df = pd.DataFrame(bets)
    n_bets = len(bets_df)
    wins = bets_df["won"].sum()
    total_profit = bets_df["profit"].sum()
    roi = total_profit / (n_bets * stake) * 100

    print(f"\n   Paris places :     {n_bets}")
    print(f"   Gagnes :           {wins} ({wins/n_bets:.1%})")
    print(f"   Profit :           {total_profit:+.2f} unites")
    print(f"   ROI :              {roi:+.1f}%")
    print(f"   Bankroll finale :  {bankroll:.2f} (depart: {initial})")
    print(f"   Edge moyen :       {bets_df['edge'].mean():.1%}")

    # Par type de pari
    print(f"\n   {'Type':<6} {'Paris':>6} {'Win%':>7} {'ROI':>8}")
    print(f"   {'-'*30}")
    for bt in ["H", "D", "A"]:
        sub = bets_df[bets_df["bet_type"] == bt]
        if len(sub) > 0:
            wr = sub["won"].mean()
            r = sub["profit"].sum() / (len(sub) * stake) * 100
            print(f"   {bt:<6} {len(sub):>6} {wr:>6.1%} {r:>+7.1f}%")

    # Graphique bankroll
    Path("charts").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bankroll_history, color="#5DCAA5", linewidth=1.5)
    ax.axhline(y=initial, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(range(len(bankroll_history)), initial, bankroll_history,
                    alpha=0.15, color="#5DCAA5" if bankroll > initial else "#F09595")
    ax.set_xlabel("Nombre de paris")
    ax.set_ylabel("Bankroll")
    ax.set_title(f"Backtest sur donnees reelles (ROI: {roi:+.1f}%)")
    plt.tight_layout()
    plt.savefig("charts/real_backtest.png", dpi=150)
    plt.close()
    print(f"\n   Graphique : charts/real_backtest.png")

    return bets_df


def main():
    print("=" * 60)
    print("ENTRAINEMENT SUR DONNEES REELLES")
    print("=" * 60)

    # 1. Charger
    df = load_data()

    # 2. Feature engineering
    df = build_features(df)

    # 3. Target
    df["target"] = df["result"].map({"H": 0, "D": 1, "A": 2})

    # 4. Split temporel
    print("\n   Split temporel...")
    train_df, test_df = temporal_split(df, test_seasons=1)

    # 5. Determiner les features
    feature_cols = get_feature_columns(train_df)

    # Supprimer les NaN dans les features
    train_clean = train_df.dropna(subset=feature_cols)
    test_clean = test_df.dropna(subset=feature_cols)

    X_train = train_clean[feature_cols]
    y_train = train_clean["target"]
    X_test = test_clean[feature_cols]
    y_test = test_clean["target"]

    # 6. Entrainer
    raw_model = train_xgboost(X_train, y_train)

    # 7. Calibrer
    calibrated = calibrate(raw_model, X_train, y_train)

    # 8. Evaluer
    metrics = evaluate(raw_model, calibrated, X_test, y_test, test_clean)

    # 9. Backtest +EV
    backtest_ev(test_clean, metrics["probabilities"], min_edge=0.03)

    # 10. Sauvegarder
    Path("model").mkdir(exist_ok=True)
    joblib.dump(calibrated, "model/xgb_calibrated_real.pkl")
    joblib.dump(feature_cols, "model/feature_cols.pkl")
    print(f"\n   Modele sauvegarde : model/xgb_calibrated_real.pkl")

    # 11. Importance des features
    print("\n   IMPORTANCE DES FEATURES :")
    importances = raw_model.feature_importances_
    for name, imp in sorted(zip(feature_cols, importances),
                            key=lambda x: x[1], reverse=True):
        bar = "#" * int(imp * 40)
        print(f"   {name:<25} {imp:.3f} {bar}")

    print("\n" + "=" * 60)
    print("Entrainement termine !")
    print("=" * 60)


if __name__ == "__main__":
    main()
