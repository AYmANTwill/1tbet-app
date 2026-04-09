"""
predictor.py - Predictions avec le modele entraine

Charge le modele XGBoost calibre et genere des predictions
sur les matchs a venir. Integre la strategie Draw-only.
"""

import numpy as np
import joblib
from pathlib import Path


# Charger le modele au demarrage du module
MODEL_PATH = Path("model/xgb_calibrated_real.pkl")
FEATURES_PATH = Path("model/feature_cols.pkl")

model = None
feature_cols = None

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)


def predict_matches(matches):
    """
    Genere des predictions pour une liste de matchs.

    Comme on n'a pas toutes les features Elo en live,
    on utilise les cotes bookmaker pour estimer les probabilites
    et le modele pour trouver les ecarts.
    """
    predictions = []

    for match in matches:
        home_odds = match.get("avg_home_odds") or match.get("best_home_odds")
        draw_odds = match.get("avg_draw_odds") or match.get("best_draw_odds")
        away_odds = match.get("avg_away_odds") or match.get("best_away_odds")

        if not all([home_odds, draw_odds, away_odds]):
            continue

        # Probabilites implicites normalisees
        imp_h = 1 / home_odds
        imp_d = 1 / draw_odds
        imp_a = 1 / away_odds
        total = imp_h + imp_d + imp_a
        norm_h = imp_h / total
        norm_d = imp_d / total
        norm_a = imp_a / total

        # Si le modele est charge, l'utiliser
        # Sinon, utiliser les probabilites implicites comme base
        if model is not None:
            # Construire les features disponibles
            # En live on n'a pas Elo ni forme, on utilise les cotes
            features = _build_live_features(match, norm_h, norm_d, norm_a)
            if features is not None:
                probs = model.predict_proba([features])[0]
                prob_h, prob_d, prob_a = probs[0], probs[1], probs[2]
            else:
                prob_h, prob_d, prob_a = norm_h, norm_d, norm_a
        else:
            # Pas de modele, on ajuste legerement les implied probs
            # Le marche sous-estime les draws en moyenne
            draw_boost = 0.03
            prob_h = norm_h - draw_boost * 0.5
            prob_d = norm_d + draw_boost
            prob_a = norm_a - draw_boost * 0.5

        # Calculer les edges
        edge_h = prob_h - (1 / home_odds)
        edge_d = prob_d - (1 / draw_odds)
        edge_a = prob_a - (1 / away_odds)

        predictions.append({
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "commence_time": match.get("commence_time"),
            "prob_home": round(float(prob_h), 4),
            "prob_draw": round(float(prob_d), 4),
            "prob_away": round(float(prob_a), 4),
            "implied_home": round(norm_h, 4),
            "implied_draw": round(norm_d, 4),
            "implied_away": round(norm_a, 4),
            "edge_home": round(float(edge_h), 4),
            "edge_draw": round(float(edge_d), 4),
            "edge_away": round(float(edge_a), 4),
            "best_home_odds": match.get("best_home_odds"),
            "best_draw_odds": match.get("best_draw_odds"),
            "best_away_odds": match.get("best_away_odds"),
            "n_bookmakers": match.get("n_bookmakers", 0),
        })

    return predictions


def _build_live_features(match, norm_h, norm_d, norm_a):
    """Construit les features pour la prediction live."""
    if feature_cols is None:
        return None

    # On remplit les features qu'on a, le reste a 0
    features = []
    for col in feature_cols:
        if col == "b365_norm_home":
            features.append(norm_h)
        elif col == "b365_norm_draw":
            features.append(norm_d)
        elif col == "b365_norm_away":
            features.append(norm_a)
        elif col == "odds_diff":
            away_odds = match.get("avg_away_odds", 3.0)
            home_odds = match.get("avg_home_odds", 2.0)
            features.append(away_odds - home_odds)
        elif col == "elo_home_prob":
            features.append(norm_h)  # Proxy
        elif col == "elo_diff":
            features.append((norm_h - norm_a) * 400)  # Proxy
        else:
            features.append(0.0)  # Forme inconnue en live

    return features


def find_ev_bets(predictions, min_edge=0.12, draw_only=True):
    """
    Filtre les paris +EV.

    draw_only=True : strategie Draw specialist (+36% ROI backteste)
    """
    bets = []

    for pred in predictions:
        match_name = f"{pred['home_team']} vs {pred['away_team']}"

        options = [
            ("H", pred["prob_home"], pred["edge_home"], pred["best_home_odds"]),
            ("D", pred["prob_draw"], pred["edge_draw"], pred["best_draw_odds"]),
            ("A", pred["prob_away"], pred["edge_away"], pred["best_away_odds"]),
        ]

        for bet_type, prob, edge, odds in options:
            if draw_only and bet_type != "D":
                continue

            if edge >= min_edge and odds and odds > 1.0:
                # Kelly fractionnel 25%
                kelly = (prob * odds - 1) / (odds - 1)
                kelly_frac = max(0, kelly * 0.25)
                kelly_frac = min(kelly_frac, 0.10)  # Max 10%

                bets.append({
                    "match": match_name,
                    "bet_type": bet_type,
                    "prob": round(float(prob), 4),
                    "odds": odds,
                    "edge": round(float(edge), 4),
                    "kelly_fraction": round(kelly_frac, 4),
                    "ev": round(float(prob * odds - 1), 4),
                    "commence_time": pred.get("commence_time"),
                })

    # Trier par edge decroissant
    bets.sort(key=lambda x: x["edge"], reverse=True)
    return bets
