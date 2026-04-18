"""
betslip_research.py - Recherche sur les paris combines (bet slips)

CONCEPT :
Si chaque pari individuel a un edge positif (+EV),
un combine de N paris independants a un edge encore plus grand.

Mathematiquement :
- 3 draws a 3.20 avec 30% de win rate chacun
- Pari simple : EV = 0.30 * 3.20 = 0.96 (perd 4%)
- Mais si notre win rate REEL est 32% grace a notre signal :
  EV = 0.32 * 3.20 = 1.024 (+2.4% par pari)
  Combine de 3 : EV = 0.32^3 * 3.20^3 = 0.0328 * 32.77 = 1.074 (+7.4%)
  L'edge se MULTIPLIE !

RISQUES :
- Variance enorme (on perd souvent, on gagne gros rarement)
- Correlation entre matchs de la meme journee
- Les bookmakers ajoutent parfois une marge supplementaire sur les combines

APPROCHES TESTEES :
1. Doubles (2 selections)
2. Triples (3 selections)
3. Quadruples (4 selections)
4. Doubles par journee (matchs du meme jour = naturellement combines)
5. Strategie mixte : 70% simples + 30% combines

Usage : python ml/betslip_research.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")


def load_and_score():
    """Charge les donnees et calcule les scores ensemble."""
    dfs = []
    for raw_dir in [Path("data/raw"), Path("data/raw_other")]:
        if not raw_dir.exists(): continue
        for f in sorted(raw_dir.glob("*.csv")):
            try:
                d = pd.read_csv(f, encoding="latin-1")
                parts = f.stem.split("_")
                d["league_code"] = parts[0]
                d["season"] = "_".join(parts[1:])
                dfs.append(d)
            except: pass
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"HomeTeam":"home_team","AwayTeam":"away_team",
                             "FTHG":"home_goals","FTAG":"away_goals","FTR":"result"})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date","result"]).sort_values("date").reset_index(drop=True)
    for c in [col for col in df.columns if any(col.startswith(p) for p in
              ["B365","BW","IW","PS","WH","VC"])]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Elo par ligue
    elos = {}
    K, HA = 25, 80
    ec = []
    for _, r in df.iterrows():
        lg = r["league_code"]
        if lg not in elos: elos[lg] = {}
        elo = elos[lg]
        ht, at = r["home_team"], r["away_team"]
        ec.append(1/(1+abs(elo.get(ht,1500)-elo.get(at,1500))/100))
        hr = elo.get(ht,1500) + HA
        ar = elo.get(at,1500)
        e = 1/(1+10**((ar-hr)/400))
        s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht] = elo.get(ht,1500) + K*(s-e)
        elo[at] = elo.get(at,1500) + K*((1-s)-(1-e))
    df["elo_close"] = ec

    # Extraire les signaux Draw
    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    signals = []
    for _, m in df.iterrows():
        odds = {c: m[c] for c in bk_d if pd.notna(m.get(c)) and m.get(c) > 1}
        if len(odds) < 3: continue
        vals = list(odds.values())
        consensus = np.mean([1/o for o in vals])
        best_bk = max(odds, key=odds.get)
        best = odds[best_bk]
        edge = consensus - 1/best
        if edge < 0.01: continue

        score = 0
        if edge >= 0.03: score += 1
        if m["elo_close"] >= 0.5: score += 1
        med = np.median(vals)
        agree = sum(1 for o in vals if abs(o-med)<0.1*med)/len(vals)
        if agree >= 0.6: score += 1
        if not best_bk.startswith("PS"): score += 1
        if best > 3.0: score += 1
        if best_bk in ["B365D","BWD","WHD","VCD"]: score += 1

        signals.append({
            "date": m["date"], "score": score, "odds": best,
            "won": m["result"]=="D", "league": m.get("league_code",""),
            "season": m.get("season",""), "edge": edge,
            "match": f"{m['home_team']} vs {m['away_team']}",
        })

    return pd.DataFrame(signals)


def simulate_singles(bets, stake=10):
    """Backtest paris simples (baseline)."""
    if bets.empty: return {"n":0,"roi":0,"profit":0,"wr":0}
    profits = bets.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
    return {"n":len(bets), "roi":profits.sum()/(len(bets)*stake)*100,
            "profit":profits.sum(), "wr":bets["won"].mean()}


def simulate_accumulators(bets, n_legs=2, stake=10, max_combos_per_day=5):
    """
    Simule des paris combines de n_legs selections.
    Groupe par date et cree des combines avec les matchs du meme jour.
    """
    bets = bets.copy()
    bets["date_str"] = bets["date"].dt.strftime("%Y-%m-%d")

    total_stake = 0
    total_return = 0
    n_slips = 0
    wins = 0
    all_slips = []

    for date, day_bets in bets.groupby("date_str"):
        if len(day_bets) < n_legs:
            continue

        # Prendre les combos possibles (limiter pour eviter explosion)
        indices = day_bets.index.tolist()
        if len(indices) > 10:
            # Trop de combos possibles, prendre les meilleurs scores
            day_bets = day_bets.nlargest(10, "score")
            indices = day_bets.index.tolist()

        combos = list(combinations(indices, n_legs))
        np.random.shuffle(combos)
        combos = combos[:max_combos_per_day]

        for combo in combos:
            legs = day_bets.loc[list(combo)]
            combined_odds = legs["odds"].prod()
            all_won = legs["won"].all()

            total_stake += stake
            if all_won:
                total_return += stake * combined_odds
                wins += 1

            n_slips += 1
            all_slips.append({
                "date": date,
                "n_legs": n_legs,
                "combined_odds": round(combined_odds, 2),
                "won": all_won,
                "profit": stake * (combined_odds - 1) if all_won else -stake,
                "matches": " | ".join(legs["match"].values),
            })

    if n_slips == 0:
        return {"n":0,"roi":0,"profit":0,"wr":0,"slips":[]}

    profit = total_return - total_stake
    roi = profit / total_stake * 100

    return {
        "n": n_slips,
        "roi": roi,
        "profit": profit,
        "wr": wins / n_slips,
        "avg_odds": np.mean([s["combined_odds"] for s in all_slips]),
        "slips": all_slips,
    }


def simulate_mixed_strategy(bets, stake_total=100, pct_singles=0.7, pct_doubles=0.2, pct_triples=0.1):
    """
    Strategie mixte : repartir le budget entre simples et combines.
    """
    stake_singles = stake_total * pct_singles
    stake_doubles = stake_total * pct_doubles
    stake_triples = stake_total * pct_triples

    n_bets = len(bets)
    stake_per_single = stake_singles / max(n_bets, 1)

    # Simples
    r_singles = simulate_singles(bets, stake=stake_per_single)

    # Doubles
    r_doubles = simulate_accumulators(bets, n_legs=2, stake=5, max_combos_per_day=3)

    # Triples
    r_triples = simulate_accumulators(bets, n_legs=3, stake=3, max_combos_per_day=2)

    total_profit = r_singles["profit"] + r_doubles["profit"] + r_triples["profit"]
    total_staked = (n_bets * stake_per_single +
                    r_doubles["n"] * 5 +
                    r_triples["n"] * 3)

    return {
        "singles": r_singles,
        "doubles": r_doubles,
        "triples": r_triples,
        "total_profit": total_profit,
        "total_roi": total_profit / total_staked * 100 if total_staked > 0 else 0,
    }


def main():
    print("=" * 60)
    print("BETSLIP RESEARCH : PARIS COMBINES")
    print("=" * 60)

    sig = load_and_score()
    print(f"\n   {len(sig)} signaux Draw extraits")

    # =========================================
    # BASELINE : PARIS SIMPLES PAR SCORE
    # =========================================
    print("\n" + "=" * 60)
    print("BASELINE : PARIS SIMPLES")
    print("=" * 60)
    for thresh in [4, 5]:
        sub = sig[sig["score"] >= thresh]
        r = simulate_singles(sub)
        print(f"   Score >= {thresh}: {r['n']} paris, Win {r['wr']:.1%}, ROI {r['roi']:+.1f}%")

    # =========================================
    # DOUBLES
    # =========================================
    print("\n" + "=" * 60)
    print("DOUBLES (2 selections)")
    print("=" * 60)
    for thresh in [4, 5]:
        sub = sig[sig["score"] >= thresh]
        r = simulate_accumulators(sub, n_legs=2, stake=10, max_combos_per_day=3)
        print(f"\n   Score >= {thresh}:")
        print(f"   Slips: {r['n']}, Win: {r['wr']:.1%}, ROI: {r['roi']:+.1f}%")
        print(f"   Avg combined odds: {r.get('avg_odds',0):.1f}")
        if r["slips"]:
            # Par saison
            sdf = pd.DataFrame(r["slips"])
            for s in sorted(set(sig["season"])):
                # Filtrer les slips dont la date correspond a la saison
                season_slips = [sl for sl in r["slips"]
                               if any(sig[(sig["match"]==m.strip()) & (sig["season"]==s)].shape[0] > 0
                                      for m in sl["matches"].split("|")[:1])]
            # Montrer quelques exemples gagnants
            wins = [s for s in r["slips"] if s["won"]][:5]
            if wins:
                print(f"   Exemples gagnants:")
                for w in wins:
                    print(f"     {w['date']}: odds {w['combined_odds']}, "
                          f"profit +{w['profit']:.0f} | {w['matches'][:60]}")

    # =========================================
    # TRIPLES
    # =========================================
    print("\n" + "=" * 60)
    print("TRIPLES (3 selections)")
    print("=" * 60)
    for thresh in [4, 5]:
        sub = sig[sig["score"] >= thresh]
        r = simulate_accumulators(sub, n_legs=3, stake=10, max_combos_per_day=2)
        print(f"\n   Score >= {thresh}:")
        print(f"   Slips: {r['n']}, Win: {r['wr']:.1%}, ROI: {r['roi']:+.1f}%")
        print(f"   Avg combined odds: {r.get('avg_odds',0):.1f}")

    # =========================================
    # QUADRUPLES
    # =========================================
    print("\n" + "=" * 60)
    print("QUADRUPLES (4 selections)")
    print("=" * 60)
    sub = sig[sig["score"] >= 4]
    r = simulate_accumulators(sub, n_legs=4, stake=10, max_combos_per_day=1)
    print(f"   Score >= 4:")
    print(f"   Slips: {r['n']}, Win: {r['wr']:.1%}, ROI: {r['roi']:+.1f}%")
    print(f"   Avg combined odds: {r.get('avg_odds',0):.1f}")

    # =========================================
    # COMPARAISON SIMPLES vs COMBINES
    # =========================================
    print("\n" + "=" * 60)
    print("COMPARAISON DIRECTE (Score >= 4)")
    print("=" * 60)
    sub = sig[sig["score"] >= 4]

    print(f"\n   {'Type':<15} {'Paris':>6} {'Win%':>7} {'ROI':>8} {'Avg odds':>9}")
    print(f"   {'-'*48}")

    r1 = simulate_singles(sub)
    print(f"   {'Simples':<15} {r1['n']:>6} {r1['wr']:>6.1%} {r1['roi']:>+7.1f}% {sub['odds'].mean():>8.2f}")

    r2 = simulate_accumulators(sub, n_legs=2, max_combos_per_day=3)
    print(f"   {'Doubles':<15} {r2['n']:>6} {r2['wr']:>6.1%} {r2['roi']:>+7.1f}% {r2.get('avg_odds',0):>8.2f}")

    r3 = simulate_accumulators(sub, n_legs=3, max_combos_per_day=2)
    print(f"   {'Triples':<15} {r3['n']:>6} {r3['wr']:>6.1%} {r3['roi']:>+7.1f}% {r3.get('avg_odds',0):>8.2f}")

    r4 = simulate_accumulators(sub, n_legs=4, max_combos_per_day=1)
    print(f"   {'Quadruples':<15} {r4['n']:>6} {r4['wr']:>6.1%} {r4['roi']:>+7.1f}% {r4.get('avg_odds',0):>8.2f}")

    # =========================================
    # STRATEGIE MIXTE OPTIMALE
    # =========================================
    print("\n" + "=" * 60)
    print("STRATEGIE MIXTE (budget reparti)")
    print("=" * 60)

    configs = [
        (1.0, 0.0, 0.0, "100% simples"),
        (0.7, 0.2, 0.1, "70/20/10"),
        (0.5, 0.3, 0.2, "50/30/20"),
        (0.3, 0.4, 0.3, "30/40/30"),
        (0.0, 0.5, 0.5, "0/50/50 (combines only)"),
    ]

    print(f"\n   {'Mix':<25} {'ROI total':>10}")
    print(f"   {'-'*37}")
    for ps, p_d, p_t, label in configs:
        r = simulate_mixed_strategy(sub, pct_singles=ps, pct_doubles=p_d, pct_triples=p_t)
        m = " $$$" if r["total_roi"] > 0 else ""
        print(f"   {label:<25} {r['total_roi']:>+9.1f}%{m}")

    # =========================================
    # CROSS-LEAGUE DOUBLES
    # =========================================
    print("\n" + "=" * 60)
    print("DOUBLES CROSS-LEAGUE (diversification)")
    print("=" * 60)
    print("   Combines entre ligues differentes pour reduire la correlation")

    sub = sig[sig["score"] >= 4].copy()
    sub["date_str"] = sub["date"].dt.strftime("%Y-%m-%d")

    cross_wins, cross_total, cross_profit = 0, 0, 0
    stake = 10

    for date, day in sub.groupby("date_str"):
        leagues = day["league"].unique()
        if len(leagues) < 2: continue

        # Prendre 1 match par ligue
        picks = []
        for lg in leagues:
            lg_bets = day[day["league"] == lg]
            best = lg_bets.nlargest(1, "score").iloc[0]
            picks.append(best)
            if len(picks) >= 2: break

        if len(picks) == 2:
            combo_odds = picks[0]["odds"] * picks[1]["odds"]
            all_won = picks[0]["won"] and picks[1]["won"]
            cross_total += 1
            if all_won:
                cross_wins += 1
                cross_profit += stake * (combo_odds - 1)
            else:
                cross_profit -= stake

    if cross_total > 0:
        cross_roi = cross_profit / (cross_total * stake) * 100
        print(f"   Slips: {cross_total}, Win: {cross_wins/cross_total:.1%}, ROI: {cross_roi:+.1f}%")
    else:
        print(f"   Pas assez de matchs cross-league")

    print("\n" + "=" * 60)
    print("CONCLUSION BETSLIP")
    print("=" * 60)
    print("   Si ROI combines > ROI simples : les combines amplifient l'edge")
    print("   Si ROI combines < ROI simples : la variance detruit l'edge")
    print("   Strategie optimale = mix simples + doubles pour equilibrer")


if __name__ == "__main__":
    main()
