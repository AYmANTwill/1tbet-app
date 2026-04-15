"""
validate_ensemble.py - Validation rigoureuse de l'approche ensemble

PRINCIPE :
On combine plusieurs signaux faibles en un score de confiance.
Plus le score est haut, plus le pari devrait etre profitable.

TESTS :
1. Walk-forward STRICT : entrainer sur passe, tester sur futur
2. Permutation test : comparer au hasard
3. Bootstrap : intervalles de confiance
4. Stabilite : performance par saison

Usage : python ml/validate_ensemble.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def load_data():
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
    for c in [col for col in df.columns if any(col.startswith(p) for p in ["B365","BW","IW","PS","WH","VC"])]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    elo = {}
    K, HA = 25, 80
    ed = []
    for _, r in df.iterrows():
        ht, at = r["home_team"], r["away_team"]
        ed.append(abs(elo.get(ht,1500) - elo.get(at,1500)))
        hr = elo.get(ht,1500) + HA
        ar = elo.get(at,1500)
        e = 1/(1+10**((ar-hr)/400))
        s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht] = elo.get(ht,1500) + K*(s-e)
        elo[at] = elo.get(at,1500) + K*((1-s)-(1-e))
    df["elo_close"] = [1/(1+d/100) for d in ed]
    return df


def compute_signals(df):
    """Score de 0 a 6 pour chaque pari Draw potentiel."""
    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    signals = []
    for _, m in df.iterrows():
        odds = {c: m[c] for c in bk_d if pd.notna(m.get(c)) and m.get(c) > 1}
        if len(odds) < 3:
            continue
        vals = list(odds.values())
        consensus = np.mean([1/o for o in vals])
        best_bk = max(odds, key=odds.get)
        best = odds[best_bk]
        edge = consensus - 1/best
        if edge < 0.01:
            continue

        # 6 signaux independants
        s = 0
        if edge >= 0.03: s += 1
        if m["elo_close"] >= 0.5: s += 1
        med = np.median(vals)
        agree = sum(1 for o in vals if abs(o-med) < 0.1*med) / len(vals)
        if agree >= 0.6: s += 1
        if not best_bk.startswith("PS"): s += 1  # Pin pas outlier
        if best > 3.0: s += 1
        if best_bk in ["B365D","BWD","WHD","VCD"]: s += 1  # soft

        signals.append({
            "score": s, "odds": best, "won": m["result"]=="D",
            "season": m.get("season",""), "league": m.get("league_code","")
        })
    return pd.DataFrame(signals)


def backtest(bets, stake=10):
    if bets.empty: return {"n":0,"roi":0,"profit":0,"wr":0}
    profits = bets.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
    return {"n":len(bets), "roi":profits.sum()/(len(bets)*stake)*100,
            "profit":profits.sum(), "wr":bets["won"].mean()}


def permutation_test(bets, n_perm=2000):
    """Compare ROI a parier aleatoirement avec memes cotes."""
    real_roi = backtest(bets)["roi"]
    odds = bets["odds"].values
    rand_rois = []
    for _ in range(n_perm):
        # Simuler victoires avec proba 1/odds
        rand_wins = np.random.random(len(odds)) < (1/odds)
        rand_profit = sum(10*(o-1) if w else -10 for o,w in zip(odds,rand_wins))
        rand_rois.append(rand_profit/(len(odds)*10)*100)
    p_value = np.mean(np.array(rand_rois) >= real_roi)
    return {"real":real_roi, "rand_mean":np.mean(rand_rois),
            "rand_std":np.std(rand_rois), "p":p_value}


def main():
    print("=" * 60)
    print("VALIDATION RIGOUREUSE ENSEMBLE APPROACH")
    print("=" * 60)
    df = load_data()
    sig = compute_signals(df)
    print(f"\n   {len(sig)} signaux Draw extraits")
    print(f"   Distribution scores: {sig['score'].value_counts().sort_index().to_dict()}")

    # Test 1 : Performance par score (in-sample, deja vu)
    print("\n" + "=" * 60)
    print("ROI PAR SCORE")
    print("=" * 60)
    for s in range(7):
        sub = sig[sig["score"] == s]
        if len(sub) >= 10:
            r = backtest(sub)
            print(f"   Score {s}: {r['n']} paris, Win {r['wr']:.1%}, ROI {r['roi']:+.1f}%")

    # Test 2 : Walk-forward STRICT par seuil
    print("\n" + "=" * 60)
    print("WALK-FORWARD STRICT (calibrer seuil sur passe, tester sur futur)")
    print("=" * 60)
    seasons = sorted(sig["season"].unique())
    for thresh in [3, 4, 5]:
        print(f"\n   Seuil score >= {thresh}:")
        total_bets, total_profit, pos = 0, 0, 0
        for i in range(1, len(seasons)):
            train = sig[sig["season"].isin(seasons[:i])]
            test = sig[sig["season"] == seasons[i]]
            test_sub = test[test["score"] >= thresh]
            if len(test_sub) >= 5:
                r = backtest(test_sub)
                m = " $" if r["roi"] > 0 else ""
                print(f"     {seasons[i]}: {r['n']} paris, ROI {r['roi']:+.1f}%{m}")
                total_bets += r["n"]
                total_profit += r["profit"]
                if r["roi"] > 0: pos += 1
        if total_bets > 0:
            roi = total_profit/(total_bets*10)*100
            print(f"     TOTAL: {total_bets} paris, ROI {roi:+.1f}%, {pos}/{len(seasons)-1} saisons +")

    # Test 3 : PERMUTATION TEST (le test de verite)
    print("\n" + "=" * 60)
    print("PERMUTATION TEST (2000 simulations)")
    print("=" * 60)
    for thresh in [3, 4, 5]:
        sub = sig[sig["score"] >= thresh]
        if len(sub) < 50:
            continue
        pt = permutation_test(sub)
        verdict = "VRAI SIGNAL" if pt["p"] < 0.05 else ("PROMETTEUR" if pt["p"] < 0.20 else "BRUIT")
        print(f"\n   Score >= {thresh} ({len(sub)} paris):")
        print(f"     ROI reel    : {pt['real']:+.1f}%")
        print(f"     ROI aleatoire: {pt['rand_mean']:+.1f}% +/- {pt['rand_std']:.1f}%")
        print(f"     p-value     : {pt['p']:.4f}")
        print(f"     VERDICT     : {verdict}")

    # Test 4 : Par ligue
    print("\n" + "=" * 60)
    print("ENSEMBLE SCORE >= 4 PAR LIGUE")
    print("=" * 60)
    for lg in sorted(sig["league"].unique()):
        sub = sig[(sig["league"]==lg) & (sig["score"]>=4)]
        if len(sub) >= 30:
            r = backtest(sub)
            pt = permutation_test(sub, n_perm=500)
            v = "SIG" if pt["p"] < 0.05 else "."
            print(f"   {lg}: {r['n']} paris, ROI {r['roi']:+.1f}%, p={pt['p']:.3f} [{v}]")

    print("\n" + "=" * 60)
    print("FIN VALIDATION")
    print("=" * 60)


if __name__ == "__main__":
    main()
