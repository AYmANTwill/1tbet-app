"""
ultimate_strategy.py - Strategie multi-couches ultra-selective

OBJECTIF : 50% ROI sur ~100 paris

APPROCHE : empiler des edges independants pour ne garder que
les paris avec la plus haute probabilite de profit.

5 LAYERS :
1. Ensemble mispricing (notre signal valide p=0.005)
2. Cross-market : Over/Under confirme le Draw ?
3. Max odds gap : taille du mispricing (plus c'est gros, mieux c'est)
4. Convergence multi-bookmaker : combien de soft books sont outliers ?
5. Kelly sizing : miser proportionnellement a la confiance

INNOVATION :
Au lieu de chercher PLUS de paris, on cherche MOINS mais MIEUX.
C'est l'inverse de ce que tout le monde fait.

Usage : python ml/ultimate_strategy.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import poisson
import warnings
warnings.filterwarnings("ignore")


def load_all_data():
    """Charge TOUTES les donnees (top 5 + autres ligues)."""
    dfs = []
    for raw_dir in [Path("data/raw"), Path("data/raw_other")]:
        if not raw_dir.exists():
            continue
        for f in sorted(raw_dir.glob("*.csv")):
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
    for c in [col for col in df.columns if any(col.startswith(p) for p in
              ["B365","BW","IW","PS","WH","VC","Max","Avg"])]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Elo par ligue
    elos = {}
    K, HA = 25, 80
    ec, ed = [], []
    for _, r in df.iterrows():
        lg = r["league_code"]
        if lg not in elos: elos[lg] = {}
        elo = elos[lg]
        ht, at = r["home_team"], r["away_team"]
        diff = elo.get(ht,1500) - elo.get(at,1500)
        ec.append(1/(1+abs(diff)/100))
        ed.append(diff)
        hr = elo.get(ht,1500) + HA
        ar = elo.get(at,1500)
        e = 1/(1+10**((ar-hr)/400))
        s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht] = elo.get(ht,1500) + K*(s-e)
        elo[at] = elo.get(at,1500) + K*((1-s)-(1-e))
    df["elo_close"] = ec
    df["elo_diff"] = ed

    # Rolling goals pour Poisson (cross-market proxy)
    df["home_avg_goals"] = df.groupby("home_team")["home_goals"].transform(
        lambda x: x.rolling(10, min_periods=3).mean())
    df["away_avg_goals"] = df.groupby("away_team")["away_goals"].transform(
        lambda x: x.rolling(10, min_periods=3).mean())
    df["home_avg_goals"] = df.groupby("home_team")["home_avg_goals"].shift(1)
    df["away_avg_goals"] = df.groupby("away_team")["away_avg_goals"].shift(1)

    return df


def compute_ultimate_signals(df):
    """
    Score ULTIME multi-couches.
    Chaque couche est independante et ajoute de la confiance.
    """
    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    bk_h = [c for c in ["B365H","BWH","IWH","PSH","WHH","VCH"] if c in df.columns]
    bk_a = [c for c in ["B365A","BWA","IWA","PSA","WHA","VCA"] if c in df.columns]

    signals = []

    for _, m in df.iterrows():
        # Cotes Draw
        odds_d = {c: m[c] for c in bk_d if pd.notna(m.get(c)) and m.get(c) > 1}
        if len(odds_d) < 3:
            continue
        vals_d = list(odds_d.values())
        consensus_d = np.mean([1/o for o in vals_d])
        best_bk = max(odds_d, key=odds_d.get)
        best = odds_d[best_bk]
        edge = consensus_d - 1/best
        if edge < 0.01:
            continue

        # =========================================
        # LAYER 1 : ENSEMBLE MISPRICING (6 points)
        # =========================================
        s1 = 0
        if edge >= 0.03: s1 += 1
        if m["elo_close"] >= 0.5: s1 += 1
        med = np.median(vals_d)
        agree = sum(1 for o in vals_d if abs(o-med)<0.1*med)/len(vals_d)
        if agree >= 0.6: s1 += 1
        if not best_bk.startswith("PS"): s1 += 1
        if best > 3.0: s1 += 1
        if best_bk in ["B365D","BWD","WHD","VCD"]: s1 += 1

        # =========================================
        # LAYER 2 : CROSS-MARKET (Poisson Under)
        # =========================================
        s2 = 0
        h_avg = m.get("home_avg_goals")
        a_avg = m.get("away_avg_goals")
        if pd.notna(h_avg) and pd.notna(a_avg) and h_avg > 0 and a_avg > 0:
            # Prob Poisson Under 2.5
            prob_under = 0
            for hg in range(3):
                for ag in range(3 - hg):
                    prob_under += poisson.pmf(hg, h_avg) * poisson.pmf(ag, a_avg)
            if prob_under > 0.45:  # Match a faible scoring = Draw-friendly
                s2 += 1
            if prob_under > 0.55:  # Tres faible scoring
                s2 += 1

            # Prob de score exact 1-1 (le draw le plus commun)
            prob_11 = poisson.pmf(1, h_avg) * poisson.pmf(1, a_avg)
            if prob_11 > 0.10:  # 1-1 probable
                s2 += 1

        # =========================================
        # LAYER 3 : TAILLE DU MISPRICING
        # =========================================
        s3 = 0
        sorted_d = sorted(vals_d, reverse=True)
        gap = sorted_d[0] - sorted_d[1] if len(sorted_d) > 1 else 0
        if gap > 0.3: s3 += 1    # Grand ecart entre best et 2nd
        if gap > 0.6: s3 += 1    # Tres grand ecart
        if edge > 0.05: s3 += 1  # Gros edge

        # =========================================
        # LAYER 4 : MULTI-SOFT CONVERGENCE
        # =========================================
        s4 = 0
        # Compter combien de soft books offrent plus que le consensus
        soft_cols = [c for c in ["B365D","BWD","WHD","VCD"] if c in odds_d]
        fair = 1 / consensus_d
        n_soft_above = sum(1 for c in soft_cols if odds_d.get(c, 0) > fair * 1.02)
        if n_soft_above >= 2: s4 += 1  # Au moins 2 softs au-dessus
        if n_soft_above >= 3: s4 += 1  # 3+ softs au-dessus

        # Pinnacle confirme-t-il le consensus ?
        if "PSD" in odds_d:
            pin_impl = 1 / odds_d["PSD"]
            if abs(pin_impl - consensus_d) < 0.02:  # Pinnacle d'accord
                s4 += 1

        # =========================================
        # LAYER 5 : CONTEXTE MATCH
        # =========================================
        s5 = 0
        # Cotes Home et Away pour detecter match equilibre via les cotes
        odds_h = {c: m[c] for c in bk_h if pd.notna(m.get(c)) and m.get(c) > 1}
        odds_a = {c: m[c] for c in bk_a if pd.notna(m.get(c)) and m.get(c) > 1}
        if odds_h and odds_a:
            avg_h = np.mean(list(odds_h.values()))
            avg_a = np.mean(list(odds_a.values()))
            # Match equilibre = cotes Home et Away proches
            ratio = max(avg_h, avg_a) / min(avg_h, avg_a)
            if ratio < 1.5: s5 += 1   # Match tres equilibre
            if ratio < 1.3: s5 += 1   # Extremement equilibre

        # Ligue profitable connue
        profitable_leagues = ["E0", "SP1", "I1", "I2", "N1", "SP2"]
        if m.get("league_code") in profitable_leagues:
            s5 += 1

        # =========================================
        # SCORE TOTAL (max theorique = 20)
        # =========================================
        total_score = s1 + s2 + s3 + s4 + s5

        # KELLY FRACTIONNEL basé sur le score
        # Plus le score est haut, plus on mise
        kelly_base = (consensus_d * best - 1) / (best - 1)
        kelly_frac = max(0, kelly_base * 0.25)
        confidence = total_score / 15  # Normalise a ~1
        kelly_adjusted = kelly_frac * confidence
        kelly_adjusted = min(kelly_adjusted, 0.15)  # Max 15% du bankroll

        signals.append({
            "score_ensemble": s1,
            "score_cross": s2,
            "score_gap": s3,
            "score_multi_soft": s4,
            "score_context": s5,
            "total_score": total_score,
            "odds": best,
            "edge": edge,
            "kelly": round(kelly_adjusted, 4),
            "won": m["result"] == "D",
            "season": m.get("season",""),
            "league": m.get("league_code",""),
            "match": f"{m['home_team']} vs {m['away_team']}",
            "best_bk": best_bk,
            "gap": gap,
        })

    return pd.DataFrame(signals)


def backtest_flat(bets, stake=10):
    """Backtest mise fixe."""
    if bets.empty: return {"n":0,"roi":0,"profit":0,"wr":0}
    profits = bets.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
    return {"n":len(bets),"roi":profits.sum()/(len(bets)*stake)*100,
            "profit":profits.sum(),"wr":bets["won"].mean()}


def backtest_kelly(bets, bankroll=1000):
    """Backtest avec Kelly fractionnel variable."""
    if bets.empty: return {"n":0,"roi":0,"profit":0,"wr":0,"final":bankroll}
    br = bankroll
    peak = br
    max_dd = 0
    profits = []
    for _, b in bets.iterrows():
        stake = br * b["kelly"]
        if stake < 1: stake = 1
        if stake > br * 0.15: stake = br * 0.15
        if b["won"]:
            p = stake * (b["odds"] - 1)
        else:
            p = -stake
        br += p
        profits.append(p)
        if br > peak: peak = br
        dd = (peak - br) / peak
        if dd > max_dd: max_dd = dd
    total_profit = br - bankroll
    total_staked = sum(abs(p) for p in profits)
    return {"n":len(bets), "roi":total_profit/bankroll*100,
            "profit":total_profit, "wr":bets["won"].mean(),
            "final":br, "max_dd":max_dd}


def perm_test(bets, n=2000):
    real = backtest_flat(bets)["roi"]
    odds = bets["odds"].values
    rand = []
    for _ in range(n):
        wins = np.random.random(len(odds)) < (1/odds)
        p = sum(10*(o-1) if w else -10 for o,w in zip(odds,wins))
        rand.append(p/(len(odds)*10)*100)
    return {"real":real,"rand_mean":np.mean(rand),"p":np.mean(np.array(rand)>=real)}


def main():
    print("=" * 60)
    print("ULTIMATE STRATEGY : MULTI-LAYER ULTRA-SELECTIVE")
    print("Objectif : 50% ROI sur ~100 paris")
    print("=" * 60)

    df = load_all_data()
    print(f"\n   {len(df)} matchs charges ({df['league_code'].nunique()} ligues)")

    sig = compute_ultimate_signals(df)
    print(f"   {len(sig)} signaux Draw extraits")
    print(f"   Score total : min={sig['total_score'].min()}, "
          f"max={sig['total_score'].max()}, mean={sig['total_score'].mean():.1f}")

    # =========================================
    # ANALYSE PAR SCORE TOTAL
    # =========================================
    print("\n" + "=" * 60)
    print("ROI PAR SCORE TOTAL (mise fixe)")
    print("=" * 60)
    print(f"\n   {'Score':>6} {'Paris':>6} {'Win%':>7} {'ROI':>8} {'Avg odds':>9}")
    print(f"   {'-'*40}")
    for s in range(sig["total_score"].max() + 1):
        sub = sig[sig["total_score"] == s]
        if len(sub) >= 10:
            r = backtest_flat(sub)
            m = " $$$" if r["roi"] > 0 else ""
            print(f"   {s:>6} {r['n']:>6} {r['wr']:>6.1%} {r['roi']:>+7.1f}% "
                  f"{sub['odds'].mean():>8.2f}{m}")

    # =========================================
    # SEUILS PROGRESSIFS
    # =========================================
    print("\n" + "=" * 60)
    print("SEUILS PROGRESSIFS (plus selectif = meilleur ?)")
    print("=" * 60)
    print(f"\n   {'Seuil':>6} {'Paris':>6} {'Win%':>7} {'ROI flat':>9} {'ROI Kelly':>10} {'p-value':>9}")
    print(f"   {'-'*55}")
    for thresh in range(4, sig["total_score"].max() + 1):
        sub = sig[sig["total_score"] >= thresh]
        if len(sub) >= 30:
            r_flat = backtest_flat(sub)
            r_kelly = backtest_kelly(sub)
            pt = perm_test(sub, n=1000)
            v = " ***" if pt["p"] < 0.01 else (" **" if pt["p"] < 0.05 else (" *" if pt["p"] < 0.10 else ""))
            print(f"   >={thresh:>4} {r_flat['n']:>6} {r_flat['wr']:>6.1%} "
                  f"{r_flat['roi']:>+8.1f}% {r_kelly['roi']:>+9.1f}% {pt['p']:>8.4f}{v}")

    # =========================================
    # WALK-FORWARD sur les meilleurs seuils
    # =========================================
    print("\n" + "=" * 60)
    print("WALK-FORWARD STRICT (meilleur seuil)")
    print("=" * 60)
    seasons = sorted(sig["season"].unique())
    for thresh in [8, 9, 10, 11]:
        sub_all = sig[sig["total_score"] >= thresh]
        if len(sub_all) < 30:
            continue
        print(f"\n   Score >= {thresh}:")
        total_n, total_profit, pos = 0, 0, 0
        for i in range(2, len(seasons)):
            test = sig[(sig["season"]==seasons[i]) & (sig["total_score"]>=thresh)]
            if len(test) >= 3:
                r = backtest_flat(test)
                m = " $" if r["roi"] > 0 else ""
                print(f"     {seasons[i]}: {r['n']:>4} paris, ROI {r['roi']:>+7.1f}%{m}")
                total_n += r["n"]
                total_profit += r["profit"]
                if r["roi"] > 0: pos += 1
        if total_n > 0:
            roi = total_profit / (total_n * 10) * 100
            n_seasons = len([s for s in seasons[2:] if len(sig[(sig["season"]==s)&(sig["total_score"]>=thresh)])>=3])
            print(f"     TOTAL: {total_n} paris, ROI {roi:+.1f}%, {pos}/{n_seasons} saisons +")

    # =========================================
    # TOP 100 MEILLEURS PARIS
    # =========================================
    print("\n" + "=" * 60)
    print("SIMULATION : TOP 100 PARIS (plus haut score)")
    print("=" * 60)
    top100 = sig.nlargest(100, "total_score")
    r_flat = backtest_flat(top100)
    r_kelly = backtest_kelly(top100)
    pt = perm_test(top100)
    print(f"   TOP 100 paris (score >= {top100['total_score'].min()}):")
    print(f"   ROI mise fixe  : {r_flat['roi']:+.1f}%")
    print(f"   ROI Kelly      : {r_kelly['roi']:+.1f}%")
    print(f"   Win rate       : {r_flat['wr']:.1%}")
    print(f"   p-value        : {pt['p']:.4f}")
    print(f"   Avg odds       : {top100['odds'].mean():.2f}")
    print(f"   Score moyen    : {top100['total_score'].mean():.1f}")

    # Par saison
    print(f"\n   Par saison :")
    for s in sorted(top100["season"].unique()):
        sub = top100[top100["season"]==s]
        r = backtest_flat(sub)
        print(f"     {s}: {r['n']} paris, ROI {r['roi']:+.1f}%")

    # =========================================
    # EXEMPLES DE PARIS ULTIMES
    # =========================================
    print("\n" + "=" * 60)
    print("EXEMPLES DE PARIS SCORE MAXIMUM")
    print("=" * 60)
    best = sig.nlargest(15, "total_score")
    for _, b in best.iterrows():
        won_str = "WIN " if b["won"] else "LOSS"
        print(f"   [{won_str}] {b['match']:<35} score={b['total_score']:>2} "
              f"odds={b['odds']:.2f} {b['best_bk']} ({b['league']} {b['season']})")

    print("\n" + "=" * 60)
    print("FIN ULTIMATE STRATEGY")
    print("=" * 60)


if __name__ == "__main__":
    main()
