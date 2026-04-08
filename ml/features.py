"""
features.py - Feature engineering pour les vraies donnees

Ce fichier reprend la logique du Repo 3 (match-predictor)
mais adapte aux vrais donnees de football-data.co.uk.

DIFFERENCES AVEC LE REPO 3 :
- On utilise de vraies cotes Bet365 et Pinnacle
- On calcule Elo par ligue (chaque ligue a ses propres ratings)
- On ajoute des features de buts concedes (defense)
- Les features sont calculees avec un strict respect du temporel
  (jamais d'information du futur)
"""

import pandas as pd
import numpy as np


class EloSystem:
    """
    Systeme Elo par ligue.
    Chaque ligue maintient ses propres ratings.
    En debut de saison, les ratings sont legerement regresses vers 1500
    (regression to the mean) pour eviter que les ecarts soient trop grands.
    """

    def __init__(self, k_factor=20, home_advantage=100):
        self.ratings = {}
        self.k = k_factor
        self.home_adv = home_advantage

    def get(self, team):
        return self.ratings.setdefault(team, 1500)

    def expected(self, home, away):
        rh = self.get(home) + self.home_adv
        ra = self.get(away)
        return 1 / (1 + 10 ** ((ra - rh) / 400))

    def update(self, home, away, result):
        exp_h = self.expected(home, away)
        exp_a = 1 - exp_h

        actual_h = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
        actual_a = 1 - actual_h

        self.ratings[home] = self.get(home) + self.k * (actual_h - exp_h)
        self.ratings[away] = self.get(away) + self.k * (actual_a - exp_a)

    def regress_to_mean(self, factor=0.8):
        """
        En debut de saison, on tire les ratings vers 1500.
        factor=0.8 signifie qu'on garde 80% de l'ecart.
        Une equipe a 1600 devient 1500 + 0.8*(1600-1500) = 1580.
        """
        for team in self.ratings:
            self.ratings[team] = 1500 + factor * (self.ratings[team] - 1500)


def compute_rolling_stats(df, team_col, stat_cols, n=5):
    """
    Calcule les moyennes glissantes des N derniers matchs.

    On utilise shift(1) pour eviter le data leakage :
    la moyenne ne contient PAS le match actuel, seulement les precedents.
    """
    result = pd.DataFrame(index=df.index)

    for col in stat_cols:
        rolled = (
            df.groupby(team_col)[col]
            .transform(lambda x: x.rolling(n, min_periods=3).mean().shift(1))
        )
        result[f"{team_col}_{col}_last{n}"] = rolled

    return result


def build_features(df):
    """
    Pipeline de feature engineering complet.

    Entree : DataFrame avec colonnes de football-data.co.uk
    Sortie : DataFrame enrichi pret pour XGBoost
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    print("   Construction des features...")

    # ============================================================
    # 1. ELO RATINGS (par ligue)
    # ============================================================
    elo_systems = {}  # Une instance par ligue
    elo_home_probs = []
    elo_diffs = []

    current_season = None

    for _, row in df.iterrows():
        league = row["league"]

        # Creer le systeme Elo pour cette ligue si necessaire
        if league not in elo_systems:
            elo_systems[league] = EloSystem(k_factor=20, home_advantage=100)

        elo = elo_systems[league]

        # Regression en debut de nouvelle saison
        if row["season"] != current_season:
            current_season = row["season"]
            elo.regress_to_mean(factor=0.8)

        # Prediction AVANT le match
        prob = elo.expected(row["home_team"], row["away_team"])
        diff = elo.get(row["home_team"]) - elo.get(row["away_team"])
        elo_home_probs.append(prob)
        elo_diffs.append(diff)

        # Mise a jour APRES le match
        elo.update(row["home_team"], row["away_team"], row["result"])

    df["elo_home_prob"] = elo_home_probs
    df["elo_diff"] = elo_diffs

    # ============================================================
    # 2. FORME RECENTE (5 derniers matchs)
    # ============================================================

    # Buts marques a domicile / a l'exterieur
    home_stats = compute_rolling_stats(df, "home_team", ["home_goals"], n=5)
    away_stats = compute_rolling_stats(df, "away_team", ["away_goals"], n=5)

    df["home_form_scored"] = home_stats["home_team_home_goals_last5"]
    df["away_form_scored"] = away_stats["away_team_away_goals_last5"]

    # Buts concedes (defense)
    home_conceded = compute_rolling_stats(df, "home_team", ["away_goals"], n=5)
    away_conceded = compute_rolling_stats(df, "away_team", ["home_goals"], n=5)

    df["home_form_conceded"] = home_conceded["home_team_away_goals_last5"]
    df["away_form_conceded"] = away_conceded["away_team_home_goals_last5"]

    # ============================================================
    # 3. COTES ET PROBABILITES IMPLICITES
    # ============================================================

    # Bet365 (bookmaker populaire)
    df["b365_impl_home"] = 1 / df["b365_home"]
    df["b365_impl_draw"] = 1 / df["b365_draw"]
    df["b365_impl_away"] = 1 / df["b365_away"]

    # Normaliser (enlever la marge)
    b365_total = df["b365_impl_home"] + df["b365_impl_draw"] + df["b365_impl_away"]
    df["b365_norm_home"] = df["b365_impl_home"] / b365_total
    df["b365_norm_draw"] = df["b365_impl_draw"] / b365_total
    df["b365_norm_away"] = df["b365_impl_away"] / b365_total

    # Pinnacle (le bookmaker le plus sharp = reference)
    if "pinnacle_home" in df.columns:
        df["pin_impl_home"] = 1 / df["pinnacle_home"]
        df["pin_impl_draw"] = 1 / df["pinnacle_draw"]
        df["pin_impl_away"] = 1 / df["pinnacle_away"]

        pin_total = df["pin_impl_home"] + df["pin_impl_draw"] + df["pin_impl_away"]
        df["pin_norm_home"] = df["pin_impl_home"] / pin_total
        df["pin_norm_draw"] = df["pin_impl_draw"] / pin_total
        df["pin_norm_away"] = df["pin_impl_away"] / pin_total

    # ============================================================
    # 4. FEATURES DERIVEES
    # ============================================================

    # Difference de cotes (signal de force relative)
    df["odds_diff"] = df["b365_away"] - df["b365_home"]

    # Marge du bookmaker (overround)
    df["overround"] = b365_total - 1

    # Difference de buts attendus (attaque - defense adverse)
    df["expected_goal_diff"] = (
        df["home_form_scored"].fillna(0) - df["away_form_conceded"].fillna(0)
    )

    # ============================================================
    # 5. NETTOYAGE
    # ============================================================

    # Supprimer les matchs sans features (debut de saison)
    df = df.dropna(subset=["home_form_scored", "away_form_scored"]).reset_index(drop=True)

    print(f"   {len(df)} matchs avec features completes")

    return df


# Colonnes utilisees par le modele
FEATURE_COLS = [
    "elo_home_prob",
    "elo_diff",
    "home_form_scored",
    "away_form_scored",
    "home_form_conceded",
    "away_form_conceded",
    "b365_norm_home",
    "b365_norm_draw",
    "b365_norm_away",
    "odds_diff",
    "expected_goal_diff",
]

# Colonnes supplementaires si Pinnacle est disponible
PINNACLE_FEATURES = [
    "pin_norm_home",
    "pin_norm_draw",
    "pin_norm_away",
]
