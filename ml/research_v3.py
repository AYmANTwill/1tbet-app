"""
research_v3.py - Methode Kaunitz : battre les bookmakers avec leurs propres chiffres

REFERENCE : Kaunitz, Zhong & Kreiner (2017)
"Beating the bookies with their own numbers"
arXiv:1710.02824 - Profitable avec de l'argent reel (+5.5% ROI)

PRINCIPE REVOLUTIONNAIRE :
On ne construit PAS de modele predictif.
On utilise la MOYENNE des cotes de tous les bookmakers comme
estimateur de la "vraie probabilite". Puis on cherche les
bookmakers individuels qui offrent des cotes significativement
au-dessus de ce consensus = cotes mispriced.

Pourquoi ca marche :
- La moyenne de 6+ bookmakers est PLUS PRECISE que chaque bookmaker individuel
- Certains bookmakers offrent des cotes trop genereuses pour attirer des clients
  ou pour equilibrer leur book
- Ces cotes mispriced representent une opportunite +EV

APPROCHES TESTEES :
1. Kaunitz pure : consensus vs bookmaker individuel
2. Pinnacle comme verite : Pinnacle (sharp) vs Bet365 (soft)
3. Over/Under 2.5 goals : marche potentiellement moins efficient
4. Combinaison ML + Kaunitz : le modele filtre, Kaunitz confirme

Usage :
    python ml/research_v3.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Charge les donnees brutes avec TOUTES les colonnes de cotes."""
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
    df = pd.concat(dfs, ignore_index=True)

    # Prep de base
    df = df.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team",
                             "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result"})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "result"]).sort_values("date").reset_index(drop=True)

    # Convertir toutes les cotes en numerique
    odds_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["B365","BW","IW","PS","WH","VC","Max","Avg"])]
    for c in odds_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"   {len(df)} matchs charges")

    # Identifier les bookmakers disponibles
    bk_home = [c for c in ["B365H","BWH","IWH","PSH","WHH","VCH"] if c in df.columns]
    bk_draw = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    bk_away = [c for c in ["B365A","BWA","IWA","PSA","WHA","VCA"] if c in df.columns]
    print(f"   Bookmakers : {len(bk_home)} (H), {len(bk_draw)} (D), {len(bk_away)} (A)")

    return df, bk_home, bk_draw, bk_away


# ============================================================
# APPROCHE 1 : METHODE KAUNITZ PURE
# ============================================================

def kaunitz_method(df, bk_home, bk_draw, bk_away, threshold=1.05):
    """
    Methode Kaunitz : trouver les cotes mispriced.

    Pour chaque match et chaque issue (H/D/A) :
    1. Calculer la probabilite consensus = moyenne des implied probs
    2. Pour chaque bookmaker, verifier si sa cote est "trop haute"
    3. Critere : odds_bookmaker > threshold / consensus_prob

    threshold = 1.05 signifie qu'on cherche des cotes > 5% au-dessus
    de la fair value estimee par le consensus.

    On parie chez le bookmaker qui offre la meilleure cote mispriced.
    """
    bets = []
    stake = 10.0

    for _, match in df.iterrows():
        for outcome, bk_cols, result_check in [
            ("H", bk_home, match["result"] == "H"),
            ("D", bk_draw, match["result"] == "D"),
            ("A", bk_away, match["result"] == "A"),
        ]:
            # Collecter les cotes disponibles
            odds_list = []
            for col in bk_cols:
                val = match.get(col)
                if pd.notna(val) and val > 1.0:
                    odds_list.append((col, val))

            if len(odds_list) < 3:
                continue

            # Probabilite consensus (moyenne des implied probs)
            implied_probs = [1/odds for _, odds in odds_list]
            consensus_prob = np.mean(implied_probs)

            # Fair odds (sans marge) = 1 / consensus_prob
            fair_odds = 1 / consensus_prob

            # Chercher les cotes mispriced (au-dessus du seuil)
            best_mispriced = None
            best_odds = 0

            for col, odds in odds_list:
                # La cote est mispriced si elle depasse fair_odds * threshold
                if odds > fair_odds * threshold and odds > best_odds:
                    best_mispriced = col
                    best_odds = odds

            if best_mispriced:
                won = result_check
                profit = stake * (best_odds - 1) if won else -stake
                edge = 1/consensus_prob - 1/best_odds

                bets.append({
                    "match": f"{match['home_team']} vs {match['away_team']}",
                    "type": outcome,
                    "bookmaker": best_mispriced,
                    "odds": best_odds,
                    "fair_odds": round(fair_odds, 2),
                    "consensus_prob": round(consensus_prob, 4),
                    "edge": round(edge, 4),
                    "won": won,
                    "profit": profit,
                    "league": match.get("league_code", ""),
                    "season": match.get("season", ""),
                })

    return pd.DataFrame(bets)


# ============================================================
# APPROCHE 2 : PINNACLE vs SOFT BOOKMAKERS
# ============================================================

def pinnacle_vs_soft(df):
    """
    Utiliser Pinnacle comme "verite" et parier chez les softs
    quand ils offrent des cotes significativement meilleures.

    Pinnacle est le bookmaker le plus "sharp" — ses cotes sont
    les plus proches de la vraie probabilite. Quand Bet365 ou
    William Hill offrent une cote bien plus haute que Pinnacle,
    c'est un signal de mispricing.
    """
    if "PSH" not in df.columns:
        print("   Pinnacle non disponible")
        return pd.DataFrame()

    soft_books = {
        "H": [c for c in ["B365H","BWH","WHH","VCH"] if c in df.columns],
        "D": [c for c in ["B365D","BWD","WHD","VCD"] if c in df.columns],
        "A": [c for c in ["B365A","BWA","WHA","VCA"] if c in df.columns],
    }
    pin_cols = {"H": "PSH", "D": "PSD", "A": "PSA"}

    bets = []
    stake = 10.0

    for _, match in df.iterrows():
        for outcome in ["H", "D", "A"]:
            pin_odds = match.get(pin_cols[outcome])
            if pd.isna(pin_odds) or pin_odds <= 1.0:
                continue

            # Prob Pinnacle (sans marge, approximation)
            pin_implied = 1 / pin_odds

            # Chercher un soft bookmaker qui offre mieux
            for col in soft_books[outcome]:
                soft_odds = match.get(col)
                if pd.isna(soft_odds) or soft_odds <= 1.0:
                    continue

                soft_implied = 1 / soft_odds
                # Le soft offre une cote plus haute = implied prob plus basse
                # Edge = pin_implied - soft_implied
                edge = pin_implied - soft_implied

                # Seuil : le soft offre > 5% de plus que Pinnacle
                if edge >= 0.03 and soft_odds > pin_odds * 1.03:
                    won = match["result"] == outcome
                    profit = stake * (soft_odds - 1) if won else -stake

                    bets.append({
                        "match": f"{match['home_team']} vs {match['away_team']}",
                        "type": outcome,
                        "bookmaker": col,
                        "odds": soft_odds,
                        "pinnacle_odds": pin_odds,
                        "edge": round(edge, 4),
                        "won": won,
                        "profit": profit,
                        "league": match.get("league_code", ""),
                        "season": match.get("season", ""),
                    })

    return pd.DataFrame(bets)


# ============================================================
# APPROCHE 3 : OVER/UNDER 2.5 GOALS (Poisson)
# ============================================================

def over_under_poisson(df):
    """
    Predire Over/Under 2.5 goals avec un modele Poisson.

    Le marche Over/Under est potentiellement moins efficient
    que le 1X2 car il necessite une modelisation differente.

    Poisson : P(k goals) = (lambda^k * e^-lambda) / k!
    P(Under 2.5) = P(0) + P(1) + P(2) pour chaque equipe
    """
    from scipy.stats import poisson

    if "B365>2.5" not in df.columns and "BbAv>2.5" not in df.columns:
        # Pas de cotes O/U dans les donnees, on utilise le total de buts
        print("   Cotes Over/Under non disponibles, utilisation du modele Poisson pur")

    # Calculer les moyennes de buts par equipe (rolling)
    df = df.copy()
    df["home_avg"] = df.groupby("home_team")["home_goals"].transform(
        lambda x: x.rolling(10, min_periods=3).mean())
    df["away_avg"] = df.groupby("away_team")["away_goals"].transform(
        lambda x: x.rolling(10, min_periods=3).mean())
    df["home_avg"] = df.groupby("home_team")["home_avg"].shift(1)
    df["away_avg"] = df.groupby("away_team")["away_avg"].shift(1)

    df = df.dropna(subset=["home_avg", "away_avg"])

    bets = []
    stake = 10.0

    # Cotes O/U si disponibles
    ou_over_col = None
    ou_under_col = None
    for c in ["B365>2.5", "BbAv>2.5"]:
        if c in df.columns:
            ou_over_col = c
            break
    for c in ["B365<2.5", "BbAv<2.5"]:
        if c in df.columns:
            ou_under_col = c
            break

    for _, match in df.iterrows():
        lambda_home = match["home_avg"]
        lambda_away = match["away_avg"]

        if lambda_home <= 0 or lambda_away <= 0:
            continue

        # Probabilite Poisson de Under 2.5
        prob_under = 0
        for h_goals in range(3):
            for a_goals in range(3 - h_goals):
                prob_under += (poisson.pmf(h_goals, lambda_home) *
                              poisson.pmf(a_goals, lambda_away))
        prob_over = 1 - prob_under

        # Total reel de buts
        total_goals = match["home_goals"] + match["away_goals"]

        # Parier si on a des cotes
        if ou_over_col and ou_under_col:
            over_odds = match.get(ou_over_col)
            under_odds = match.get(ou_under_col)

            if pd.notna(over_odds) and over_odds > 1.0:
                implied_over = 1 / over_odds
                edge_over = prob_over - implied_over
                if edge_over >= 0.05:
                    won = total_goals > 2.5
                    bets.append({
                        "match": f"{match['home_team']} vs {match['away_team']}",
                        "type": "Over 2.5",
                        "odds": over_odds,
                        "prob": round(prob_over, 4),
                        "edge": round(edge_over, 4),
                        "won": won,
                        "profit": stake * (over_odds - 1) if won else -stake,
                        "league": match.get("league_code", ""),
                        "season": match.get("season", ""),
                    })

            if pd.notna(under_odds) and under_odds > 1.0:
                implied_under = 1 / under_odds
                edge_under = prob_under - implied_under
                if edge_under >= 0.05:
                    won = total_goals < 2.5
                    bets.append({
                        "match": f"{match['home_team']} vs {match['away_team']}",
                        "type": "Under 2.5",
                        "odds": under_odds,
                        "prob": round(prob_under, 4),
                        "edge": round(edge_under, 4),
                        "won": won,
                        "profit": stake * (under_odds - 1) if won else -stake,
                        "league": match.get("league_code", ""),
                        "season": match.get("season", ""),
                    })

    return pd.DataFrame(bets)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_strategy(bets_df, name):
    """Evaluation walk-forward par saison."""
    if bets_df.empty:
        print(f"\n   {name} : aucun pari")
        return

    print(f"\n   {name}")
    print(f"   {'='*60}")

    seasons = sorted(bets_df["season"].unique())
    stake = 10.0

    print(f"   {'Season':<15} {'Paris':>6} {'Win%':>7} {'ROI':>8} {'Profit':>10} {'Avg odds':>9}")
    print(f"   {'-'*57}")

    total_n = 0
    total_profit = 0
    season_pos = 0

    for season in seasons:
        s = bets_df[bets_df["season"] == season]
        n = len(s)
        if n == 0:
            continue
        wins = s["won"].sum()
        profit = s["profit"].sum()
        roi = profit / (n * stake) * 100
        avg_odds = s["odds"].mean()

        marker = " $$$" if roi > 0 and n >= 10 else ""
        if roi > 0 and n >= 10:
            season_pos += 1

        print(f"   {season:<15} {n:>6} {wins/n:>6.1%} {roi:>+7.1f}% {profit:>+9.1f} {avg_odds:>8.2f}{marker}")

        total_n += n
        total_profit += profit

    total_roi = total_profit / (total_n * stake) * 100 if total_n > 0 else 0
    print(f"   {'-'*57}")
    print(f"   {'TOTAL':<15} {total_n:>6} {'':>7} {total_roi:>+7.1f}% {total_profit:>+9.1f}")
    print(f"   Saisons positives : {season_pos}/{len(seasons)}")

    # Par type de pari
    if "type" in bets_df.columns:
        print(f"\n   Par type :")
        for bt in sorted(bets_df["type"].unique()):
            sub = bets_df[bets_df["type"] == bt]
            if len(sub) > 0:
                r = sub["profit"].sum() / (len(sub) * stake) * 100
                print(f"   {bt:<10} : {len(sub)} paris, ROI {r:+.1f}%")

    # Par ligue
    if "league" in bets_df.columns:
        print(f"\n   Par ligue :")
        for lg in sorted(bets_df["league"].unique()):
            sub = bets_df[bets_df["league"] == lg]
            if len(sub) > 0:
                r = sub["profit"].sum() / (len(sub) * stake) * 100
                print(f"   {lg:<10} : {len(sub)} paris, ROI {r:+.1f}%")

    return total_roi


def main():
    print("=" * 60)
    print("RECHERCHE V3 : METHODES BASEES SUR LE MARCHE")
    print("(Kaunitz et al. 2017 + Pinnacle + Poisson)")
    print("=" * 60)

    df, bk_home, bk_draw, bk_away = load_data()

    # ============================================================
    # APPROCHE 1 : KAUNITZ (consensus mispricing)
    # ============================================================
    print("\n" + "=" * 60)
    print("APPROCHE 1 : METHODE KAUNITZ (consensus vs outlier)")
    print("=" * 60)

    for threshold in [1.03, 1.05, 1.08, 1.10]:
        bets = kaunitz_method(df, bk_home, bk_draw, bk_away, threshold=threshold)
        if not bets.empty:
            evaluate_strategy(bets, f"Kaunitz (seuil={threshold})")

    # ============================================================
    # APPROCHE 2 : PINNACLE vs SOFT
    # ============================================================
    print("\n" + "=" * 60)
    print("APPROCHE 2 : PINNACLE vs BOOKMAKERS SOFT")
    print("=" * 60)

    bets_pin = pinnacle_vs_soft(df)
    if not bets_pin.empty:
        evaluate_strategy(bets_pin, "Pinnacle vs Soft (edge > 3%)")

    # ============================================================
    # APPROCHE 3 : OVER/UNDER POISSON
    # ============================================================
    print("\n" + "=" * 60)
    print("APPROCHE 3 : OVER/UNDER 2.5 (Poisson)")
    print("=" * 60)

    bets_ou = over_under_poisson(df)
    if not bets_ou.empty:
        evaluate_strategy(bets_ou, "Over/Under Poisson (edge > 5%)")

    # ============================================================
    # APPROCHE 4 : KAUNITZ + ML FILTER
    # ============================================================
    print("\n" + "=" * 60)
    print("APPROCHE 4 : KAUNITZ + FILTRE CONTEXTUEL")
    print("=" * 60)
    print("   (Kaunitz pour trouver les cotes, Elo pour filtrer)")

    # Ajouter Elo au dataframe
    elo = {}
    K, HA = 25, 80
    elo_close = []
    for _, r in df.iterrows():
        ht, at = r["home_team"], r["away_team"]
        diff = abs(elo.get(ht, 1500) - elo.get(at, 1500))
        elo_close.append(1 / (1 + diff / 100))
        hr = elo.get(ht, 1500) + HA
        ar = elo.get(at, 1500)
        e = 1 / (1 + 10**((ar-hr)/400))
        s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht] = elo.get(ht,1500) + K*(s-e)
        elo[at] = elo.get(at,1500) + K*((1-s)-(1-e))
    df["elo_closeness"] = elo_close

    # Kaunitz seulement sur matchs equilibres (Elo closeness > 0.6)
    df_balanced = df[df["elo_closeness"] > 0.6]
    print(f"   Matchs equilibres (Elo closeness > 0.6) : {len(df_balanced)}")

    bets_balanced = kaunitz_method(df_balanced, bk_home, bk_draw, bk_away, threshold=1.05)
    if not bets_balanced.empty:
        evaluate_strategy(bets_balanced, "Kaunitz + matchs equilibres")

    # Kaunitz seulement sur Draw (matchs equilibres)
    if not bets_balanced.empty:
        bets_draw = bets_balanced[bets_balanced["type"] == "D"]
        if not bets_draw.empty:
            evaluate_strategy(bets_draw, "Kaunitz Draw + matchs equilibres")

    print("\n" + "=" * 60)
    print("RECHERCHE V3 TERMINEE")
    print("=" * 60)


if __name__ == "__main__":
    main()
