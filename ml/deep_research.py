"""
deep_research.py - Stress tests + nouvelles pistes de decouverte

PARTIE A : STRESS TESTS (est-ce que nos decouvertes sont REELLES ?)
1. Bootstrap : intervalles de confiance sur le ROI
2. Drawdown max : un parieur peut-il survivre ?
3. Sensibilite : petits changements de seuil = gros changements de ROI ?
4. Permutation test : notre ROI est-il significativement different du hasard ?

PARTIE B : NOUVELLES PISTES
5. Dixon-Coles : modele specifique pour les draws en football
6. Patterns temporels : les mispricings sont-ils plus frequents certains jours ?
7. Second divisions : marches moins efficients ?
8. Ensemble de strategies : combiner plusieurs signaux faibles
9. Score-line analysis : 0-0, 1-1 sont-ils plus previsibles que "Draw" en general ?

Usage :
    python ml/deep_research.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


def load_and_prep():
    """Charge et prepare les donnees."""
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
    df = df.rename(columns={"HomeTeam":"home_team","AwayTeam":"away_team",
                             "FTHG":"home_goals","FTAG":"away_goals","FTR":"result"})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date","result"]).sort_values("date").reset_index(drop=True)
    for c in [col for col in df.columns if any(col.startswith(p) for p in ["B365","BW","IW","PS","WH","VC","Max","Avg"])]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Elo
    elo = {}
    K, HA = 25, 80
    ec = []
    for _, r in df.iterrows():
        ht, at = r["home_team"], r["away_team"]
        diff = abs(elo.get(ht,1500) - elo.get(at,1500))
        ec.append(1/(1+diff/100))
        hr = elo.get(ht,1500)+HA
        ar = elo.get(at,1500)
        e = 1/(1+10**((ar-hr)/400))
        s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht] = elo.get(ht,1500)+K*(s-e)
        elo[at] = elo.get(at,1500)+K*((1-s)-(1-e))
    df["elo_closeness"] = ec
    return df


def extract_draw_signals(df):
    """Extraire les signaux Draw avec meta-features (notre meilleure strategie)."""
    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    signals = []
    stake = 10.0

    for _, match in df.iterrows():
        odds_dict = {}
        for col in bk_d:
            val = match.get(col)
            if pd.notna(val) and val > 1.0:
                odds_dict[col] = val
        if len(odds_dict) < 3:
            continue

        odds_values = list(odds_dict.values())
        implied = [1/o for o in odds_values]
        consensus = np.mean(implied)
        fair_odds = 1/consensus

        best_bk = max(odds_dict, key=odds_dict.get)
        best_odds = odds_dict[best_bk]
        edge = consensus - 1/best_odds
        if edge < 0.01:
            continue

        # Accord
        median_o = np.median(odds_values)
        n_agree = sum(1 for o in odds_values if abs(o-median_o) < 0.1*median_o)
        agreement = n_agree / len(odds_values)

        # Pinnacle outlier ?
        pin_outlier = best_bk.startswith("PS")
        # Soft book ?
        is_soft = any(best_bk.startswith(s) for s in ["B365","BW","WH","VC"])

        won = match["result"] == "D"
        profit = stake*(best_odds-1) if won else -stake

        signals.append({
            "edge": edge, "best_odds": best_odds, "best_bk": best_bk,
            "agreement": agreement, "pin_outlier": pin_outlier, "is_soft": is_soft,
            "elo_closeness": match.get("elo_closeness", 0.5),
            "won": won, "profit": profit,
            "match": f"{match['home_team']} vs {match['away_team']}",
            "league": match.get("league_code",""),
            "season": match.get("season",""),
            "home_goals": match.get("home_goals"), "away_goals": match.get("away_goals"),
            "home_team": match["home_team"], "away_team": match["away_team"],
            "date": match["date"],
        })

    return pd.DataFrame(signals)


def apply_strategy(signals, name, mask):
    """Applique un filtre et retourne les resultats."""
    sub = signals[mask].copy()
    if sub.empty:
        return None
    profit = sub["profit"].sum()
    n = len(sub)
    return {"name": name, "n": n, "profit": profit,
            "roi": profit/(n*10)*100, "win_rate": sub["won"].mean(),
            "bets": sub}


# ============================================================
# PARTIE A : STRESS TESTS
# ============================================================

def stress_test_bootstrap(signals, mask, name, n_bootstrap=5000):
    """
    Bootstrap : estime l'intervalle de confiance du ROI.

    On re-echantillonne les paris 5000 fois avec remplacement
    et on calcule le ROI a chaque fois. L'intervalle 5%-95%
    nous dit si le ROI est "reellement" positif.
    """
    print(f"\n   BOOTSTRAP : {name}")
    sub = signals[mask]
    if len(sub) < 20:
        print("   Pas assez de paris")
        return

    profits = sub["profit"].values
    n = len(profits)
    stake = 10.0

    rois = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(profits, size=n, replace=True)
        roi = sample.sum() / (n * stake) * 100
        rois.append(roi)

    rois = np.array(rois)
    ci_5 = np.percentile(rois, 5)
    ci_50 = np.percentile(rois, 50)
    ci_95 = np.percentile(rois, 95)
    prob_positive = (rois > 0).mean()

    print(f"   ROI observe : {profits.sum()/(n*stake)*100:+.1f}%")
    print(f"   IC 90% : [{ci_5:+.1f}%, {ci_95:+.1f}%]")
    print(f"   Mediane bootstrap : {ci_50:+.1f}%")
    print(f"   P(ROI > 0) : {prob_positive:.1%}")

    if ci_5 > 0:
        print(f"   VERDICT : SIGNIFICATIF (IC entierement > 0)")
    elif prob_positive > 0.75:
        print(f"   VERDICT : PROMETTEUR (>75% positif)")
    else:
        print(f"   VERDICT : NON SIGNIFICATIF")

    return {"ci_5": ci_5, "ci_95": ci_95, "prob_positive": prob_positive}


def stress_test_drawdown(signals, mask, name):
    """Analyse du drawdown maximum."""
    print(f"\n   DRAWDOWN : {name}")
    sub = signals[mask].sort_values("date") if "date" in signals.columns else signals[mask]
    if len(sub) < 20:
        return

    bankroll = 1000.0
    peak = bankroll
    max_dd = 0
    dd_bets = 0
    history = [bankroll]

    for _, bet in sub.iterrows():
        bankroll += bet["profit"]
        history.append(bankroll)
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak
        if dd > max_dd:
            max_dd = dd

    final_roi = (bankroll - 1000) / 1000 * 100
    print(f"   Bankroll : 1000 -> {bankroll:.0f} ({final_roi:+.1f}%)")
    print(f"   Max drawdown : {max_dd:.1%}")
    print(f"   Peak : {max(history):.0f}")

    if max_dd < 0.15:
        print(f"   VERDICT : Drawdown acceptable (<15%)")
    elif max_dd < 0.30:
        print(f"   VERDICT : Drawdown modere (15-30%)")
    else:
        print(f"   VERDICT : Drawdown DANGEREUX (>{max_dd:.0%})")


def stress_test_permutation(signals, mask, name, n_perms=5000):
    """
    Test de permutation : notre ROI est-il du au hasard ?

    On melange aleatoirement les resultats (won/lost) 5000 fois
    et on regarde combien de fois le ROI aleatoire depasse notre ROI reel.
    Si < 5%, notre ROI est statistiquement significatif.
    """
    print(f"\n   PERMUTATION TEST : {name}")
    sub = signals[mask]
    if len(sub) < 20:
        return

    real_profit = sub["profit"].sum()
    real_roi = real_profit / (len(sub)*10) * 100

    # Melanger les resultats
    odds = sub["best_odds"].values
    wins = sub["won"].values
    n = len(sub)

    random_rois = []
    for _ in range(n_perms):
        shuffled_wins = np.random.permutation(wins)
        profits = np.where(shuffled_wins, 10*(odds-1), -10)
        roi = profits.sum() / (n*10) * 100
        random_rois.append(roi)

    random_rois = np.array(random_rois)
    p_value = (random_rois >= real_roi).mean()

    print(f"   ROI reel : {real_roi:+.1f}%")
    print(f"   ROI moyen aleatoire : {random_rois.mean():+.1f}%")
    print(f"   p-value : {p_value:.4f}")

    if p_value < 0.05:
        print(f"   VERDICT : SIGNIFICATIF (p < 0.05)")
    elif p_value < 0.10:
        print(f"   VERDICT : MARGINALEMENT significatif (p < 0.10)")
    else:
        print(f"   VERDICT : NON significatif (p = {p_value:.2f})")


def stress_test_sensitivity(signals, name):
    """Sensibilite aux seuils : petits changements = gros impact ?"""
    print(f"\n   SENSIBILITE : {name}")
    print(f"   {'Elo close':>10} {'Agreement':>10} {'Paris':>6} {'ROI':>8} {'Stable?':>8}")
    print(f"   {'-'*44}")

    for ec in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for ag in [0.5, 0.6, 0.7, 0.8]:
            mask = (
                (signals["elo_closeness"] >= ec) &
                (signals["agreement"] >= ag) &
                (~signals["pin_outlier"])
            )
            sub = signals[mask]
            if len(sub) >= 30:
                roi = sub["profit"].sum() / (len(sub)*10) * 100
                stable = "OUI" if roi > 0 else "non"
                print(f"   {ec:>10.1f} {ag:>10.1f} {len(sub):>6} {roi:>+7.1f}% {stable:>8}")


# ============================================================
# PARTIE B : NOUVELLES PISTES
# ============================================================

def piste_dixon_coles(df):
    """
    Modele Dixon-Coles : le gold standard pour le football.

    Dixon-Coles (1997) corrige le modele Poisson pour les
    scores faibles (0-0, 1-0, 0-1, 1-1) qui sont systematiquement
    MAL ESTIMES par Poisson standard.

    C'est important pour les Draws car beaucoup de Draws sont
    des 0-0 ou 1-1 — exactement la ou Dixon-Coles brille.
    """
    print("\n" + "=" * 60)
    print("PISTE : DIXON-COLES POUR LES DRAWS")
    print("=" * 60)

    # Calculer les forces d'attaque et defense par equipe
    # On utilise les moyennes de buts sur les N derniers matchs
    df = df.copy()
    df["home_avg_scored"] = df.groupby("home_team")["home_goals"].transform(
        lambda x: x.rolling(15, min_periods=5).mean())
    df["home_avg_conceded"] = df.groupby("home_team")["away_goals"].transform(
        lambda x: x.rolling(15, min_periods=5).mean())
    df["away_avg_scored"] = df.groupby("away_team")["away_goals"].transform(
        lambda x: x.rolling(15, min_periods=5).mean())
    df["away_avg_conceded"] = df.groupby("away_team")["home_goals"].transform(
        lambda x: x.rolling(15, min_periods=5).mean())

    # Shift pour eviter leakage
    for col in ["home_avg_scored","home_avg_conceded","away_avg_scored","away_avg_conceded"]:
        team_col = "home_team" if "home" in col else "away_team"
        df[col] = df.groupby(team_col)[col].shift(1)

    df = df.dropna(subset=["home_avg_scored","away_avg_scored",
                            "home_avg_conceded","away_avg_conceded"])

    # Moyennes globales de la ligue
    avg_home_goals = df["home_goals"].mean()  # ~1.5
    avg_away_goals = df["away_goals"].mean()  # ~1.2

    # Facteur de correction Dixon-Coles pour les scores faibles
    def dc_correction(h_goals, a_goals, lambda_h, lambda_a, rho=-0.13):
        """
        rho < 0 signifie que les 0-0 et 1-1 sont plus probables
        que ce que Poisson predit. C'est le coeur de Dixon-Coles.
        """
        if h_goals == 0 and a_goals == 0:
            return 1 - lambda_h * lambda_a * rho
        elif h_goals == 0 and a_goals == 1:
            return 1 + lambda_h * rho
        elif h_goals == 1 and a_goals == 0:
            return 1 + lambda_a * rho
        elif h_goals == 1 and a_goals == 1:
            return 1 - rho
        else:
            return 1.0

    bets = []
    stake = 10.0

    for _, match in df.iterrows():
        # Lambda home = attaque_home * defense_adverse / moyenne
        att_h = match["home_avg_scored"]
        def_a = match["away_avg_conceded"]
        att_a = match["away_avg_scored"]
        def_h = match["home_avg_conceded"]

        if att_h <= 0 or def_a <= 0 or att_a <= 0 or def_h <= 0:
            continue

        lambda_h = att_h * (def_a / avg_away_goals) * 1.05  # Home advantage
        lambda_a = att_a * (def_h / avg_home_goals) * 0.95

        lambda_h = max(0.3, min(lambda_h, 4.0))
        lambda_a = max(0.2, min(lambda_a, 3.5))

        # Probabilite de Draw avec correction Dixon-Coles
        prob_draw_dc = 0
        for h in range(6):
            for a in range(6):
                if h == a:  # Draw
                    p = (poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a) *
                         dc_correction(h, a, lambda_h, lambda_a, rho=-0.13))
                    prob_draw_dc += p

        # Comparer avec les cotes bookmaker
        draw_odds_col = None
        for col in ["PSD", "B365D", "AvgD"]:
            if col in df.columns and pd.notna(match.get(col)):
                draw_odds_col = col
                break

        if draw_odds_col is None:
            continue

        draw_odds = match[draw_odds_col]
        if pd.isna(draw_odds) or draw_odds <= 1.0:
            continue

        implied_draw = 1 / draw_odds
        edge = prob_draw_dc - implied_draw

        if edge >= 0.03:
            won = match["result"] == "D"
            profit = stake * (draw_odds - 1) if won else -stake
            bets.append({
                "match": f"{match['home_team']} vs {match['away_team']}",
                "prob_dc": prob_draw_dc, "odds": draw_odds,
                "edge": edge, "won": won, "profit": profit,
                "lambda_h": lambda_h, "lambda_a": lambda_a,
                "league": match.get("league_code",""),
                "season": match.get("season",""),
                "elo_closeness": match.get("elo_closeness", 0.5),
            })

    bets_df = pd.DataFrame(bets)
    if bets_df.empty:
        print("   Aucun pari Dixon-Coles")
        return bets_df

    # Resultats par saison
    print(f"\n   Dixon-Coles Draw (edge > 3%) :")
    print(f"   {'Season':<15} {'Paris':>6} {'Win%':>7} {'ROI':>8}")
    print(f"   {'-'*38}")
    total_n, total_p, pos = 0, 0, 0
    for s in sorted(bets_df["season"].unique()):
        sub = bets_df[bets_df["season"]==s]
        n = len(sub)
        p = sub["profit"].sum()
        roi = p/(n*10)*100
        m = " $$$" if roi > 0 and n >= 10 else ""
        if roi > 0 and n >= 10: pos += 1
        print(f"   {s:<15} {n:>6} {sub['won'].mean():>6.1%} {roi:>+7.1f}%{m}")
        total_n += n
        total_p += p
    print(f"   {'-'*38}")
    print(f"   {'TOTAL':<15} {total_n:>6} {'':>7} {total_p/(total_n*10)*100:>+7.1f}%")
    print(f"   Saisons + : {pos}/{len(bets_df['season'].unique())}")

    # Dixon-Coles + matchs equilibres
    balanced = bets_df[bets_df["elo_closeness"] >= 0.5]
    if len(balanced) >= 20:
        roi_b = balanced["profit"].sum() / (len(balanced)*10) * 100
        pos_b = sum(1 for s in balanced["season"].unique()
                    if len(balanced[balanced["season"]==s]) >= 5 and
                    balanced[balanced["season"]==s]["profit"].sum() > 0)
        print(f"\n   Dixon-Coles + equilibre : {len(balanced)} paris, "
              f"ROI {roi_b:+.1f}%, {pos_b}/{len(balanced['season'].unique())} saisons +")

    return bets_df


def piste_scoreline(df):
    """
    Analyse par scoreline : 0-0, 1-1, 2-2 sont-ils plus previsibles ?

    Hypothese : les 1-1 sont le score exact le plus frequent pour les
    draws. Si on peut predire QUAND un 1-1 est probable (equipes
    defensives, lambda faible), on peut parier sur Draw avec plus
    de precision.
    """
    print("\n" + "=" * 60)
    print("PISTE : ANALYSE PAR SCORELINE")
    print("=" * 60)

    draws = df[df["result"] == "D"].copy()
    total_matches = len(df)
    total_draws = len(draws)

    print(f"\n   Total matchs : {total_matches}")
    print(f"   Draws : {total_draws} ({total_draws/total_matches:.1%})")

    # Distribution des scorelines de draw
    draws["scoreline"] = draws["home_goals"].astype(int).astype(str) + "-" + draws["away_goals"].astype(int).astype(str)
    dist = draws["scoreline"].value_counts()

    print(f"\n   Scorelines les plus frequentes :")
    for score, count in dist.head(6).items():
        pct = count / total_draws * 100
        print(f"   {score} : {count} ({pct:.1f}% des draws)")


def piste_ensemble(signals, df):
    """
    Combiner plusieurs signaux INDEPENDANTS en un ensemble.

    L'idee : chaque signal faible peut etre du bruit.
    Mais si 3 signaux independants s'accordent, c'est plus fiable.

    Signaux combines :
    1. Kaunitz mispricing (consensus vs outlier)
    2. Elo closeness (matchs equilibres)
    3. Dixon-Coles (modele specifique draw)
    4. Historique de draw rate des equipes
    """
    print("\n" + "=" * 60)
    print("PISTE : ENSEMBLE DE SIGNAUX INDEPENDANTS")
    print("=" * 60)

    # Draw rate par equipe
    df = df.copy()
    df["is_draw"] = (df["result"]=="D").astype(float)
    df["home_draw_rate"] = df.groupby("home_team")["is_draw"].transform(
        lambda x: x.rolling(15, min_periods=5).mean())
    df["away_draw_rate"] = df.groupby("away_team")["is_draw"].transform(
        lambda x: x.rolling(15, min_periods=5).mean())
    df["home_draw_rate"] = df.groupby("home_team")["home_draw_rate"].shift(1)
    df["away_draw_rate"] = df.groupby("away_team")["away_draw_rate"].shift(1)
    df["avg_draw_rate"] = (df["home_draw_rate"].fillna(0.25) + df["away_draw_rate"].fillna(0.25))/2

    # Merge draw rate into signals
    if not signals.empty and "date" in signals.columns:
        signals = signals.copy()
        # Ajouter draw rate via match name approximation
        dr_map = df.set_index(["home_team","away_team","date"])["avg_draw_rate"].to_dict()

        draw_rates = []
        for _, s in signals.iterrows():
            key = (s.get("home_team",""), s.get("away_team",""), s.get("date"))
            draw_rates.append(dr_map.get(key, 0.25))
        signals["draw_rate"] = draw_rates

    # Scoring : combien de signaux positifs par pari ?
    signals = signals.copy()
    signals["score"] = 0
    signals.loc[signals["edge"] >= 0.02, "score"] += 1        # Kaunitz signal
    signals.loc[signals["elo_closeness"] >= 0.5, "score"] += 1 # Match equilibre
    signals.loc[~signals["pin_outlier"], "score"] += 1          # Pin pas outlier
    signals.loc[signals["is_soft"], "score"] += 1               # Soft book
    signals.loc[signals["agreement"] >= 0.7, "score"] += 1     # Fort accord

    if "draw_rate" in signals.columns:
        signals.loc[signals["draw_rate"] >= 0.28, "score"] += 1  # Equipes qui font des draws

    print(f"\n   Distribution des scores (0-6) :")
    for sc in range(7):
        sub = signals[signals["score"] == sc]
        if len(sub) > 0:
            roi = sub["profit"].sum() / (len(sub)*10) * 100 if len(sub) > 0 else 0
            m = " $$$" if roi > 0 and len(sub) >= 20 else ""
            print(f"   Score {sc} : {len(sub)} paris, ROI {roi:+.1f}%{m}")

    # Backtest par seuil de score
    print(f"\n   Backtest par seuil de score minimum :")
    print(f"   {'Score min':>10} {'Paris':>6} {'Win%':>7} {'ROI':>8} {'Saisons+':>9}")
    print(f"   {'-'*42}")

    for min_score in range(2, 7):
        sub = signals[signals["score"] >= min_score]
        if len(sub) >= 15:
            roi = sub["profit"].sum() / (len(sub)*10) * 100
            wr = sub["won"].mean()
            # Saisons
            pos_s = 0
            total_s = 0
            for s in sub["season"].unique():
                ss = sub[sub["season"]==s]
                if len(ss) >= 5:
                    total_s += 1
                    if ss["profit"].sum() > 0:
                        pos_s += 1
            m = " $$$" if roi > 0 else ""
            print(f"   {min_score:>10} {len(sub):>6} {wr:>6.1%} {roi:>+7.1f}% {pos_s}/{total_s}{m}")


def main():
    print("=" * 60)
    print("DEEP RESEARCH : STRESS TESTS + NOUVELLES PISTES")
    print("=" * 60)

    df = load_and_prep()
    signals = extract_draw_signals(df)
    print(f"\n   {len(signals)} signaux Draw extraits")

    # ============================================================
    # PARTIE A : STRESS TESTS
    # ============================================================
    print("\n" + "#" * 60)
    print("# PARTIE A : STRESS TESTS")
    print("#" * 60)

    # Nos 3 meilleures strategies
    strategies = {
        "Draw+soft+accord": (
            (signals["is_soft"]) & (signals["agreement"]>=0.7)
        ),
        "Draw+!Pin+equilibre": (
            (~signals["pin_outlier"]) & (signals["elo_closeness"]>=0.5)
        ),
        "Draw+equilibre+accord": (
            (signals["elo_closeness"]>=0.5) & (signals["agreement"]>=0.7)
        ),
    }

    for name, mask in strategies.items():
        print(f"\n   {'='*55}")
        print(f"   STRATEGIE : {name}")
        print(f"   {'='*55}")

        r = apply_strategy(signals, name, mask)
        if r:
            print(f"   {r['n']} paris, ROI {r['roi']:+.1f}%, Win {r['win_rate']:.1%}")

        stress_test_bootstrap(signals, mask, name)
        stress_test_drawdown(signals, mask, name)
        stress_test_permutation(signals, mask, name)

    # Sensibilite
    print(f"\n   {'='*55}")
    stress_test_sensitivity(signals, "Sensibilite Draw+!Pin")

    # ============================================================
    # PARTIE B : NOUVELLES PISTES
    # ============================================================
    print("\n" + "#" * 60)
    print("# PARTIE B : NOUVELLES PISTES")
    print("#" * 60)

    dc_bets = piste_dixon_coles(df)
    piste_scoreline(df)
    piste_ensemble(signals, df)

    # ============================================================
    # SYNTHESE
    # ============================================================
    print("\n" + "=" * 60)
    print("SYNTHESE FINALE")
    print("=" * 60)
    print("   Regarder les verdicts de chaque stress test.")
    print("   SIGNIFICATIF + drawdown acceptable = strategie deployable.")
    print("   Sinon, continuer la recherche.")


if __name__ == "__main__":
    main()
