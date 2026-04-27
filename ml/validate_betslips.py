"""
validate_betslips.py - Permutation test sur les bet slips

Est-ce que le +50.7% ROI des triples est un VRAI signal ?
On compare a des combinaisons aleatoires avec les memes cotes.

Usage : python ml/validate_betslips.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")


def load_and_score():
    """Charge et score les signaux."""
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

    elos = {}
    K, HA = 25, 80
    ec = []
    for _, r in df.iterrows():
        lg = r["league_code"]
        if lg not in elos: elos[lg] = {}
        elo = elos[lg]
        ht, at = r["home_team"], r["away_team"]
        ec.append(1/(1+abs(elo.get(ht,1500)-elo.get(at,1500))/100))
        hr = elo.get(ht,1500) + HA; ar = elo.get(at,1500)
        e = 1/(1+10**((ar-hr)/400))
        s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht] = elo.get(ht,1500) + K*(s-e)
        elo[at] = elo.get(at,1500) + K*((1-s)-(1-e))
    df["elo_close"] = ec

    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    sigs = []
    for _, m in df.iterrows():
        odds = {c: m[c] for c in bk_d if pd.notna(m.get(c)) and m.get(c) > 1}
        if len(odds) < 3: continue
        vals = list(odds.values())
        consensus = np.mean([1/o for o in vals])
        best_bk = max(odds, key=odds.get)
        best = odds[best_bk]
        edge = consensus - 1/best
        if edge < 0.01: continue
        sc = 0
        if edge >= 0.03: sc += 1
        if m["elo_close"] >= 0.5: sc += 1
        med = np.median(vals)
        if sum(1 for o in vals if abs(o-med)<0.1*med)/len(vals) >= 0.6: sc += 1
        if not best_bk.startswith("PS"): sc += 1
        if best > 3.0: sc += 1
        if best_bk in ["B365D","BWD","WHD","VCD"]: sc += 1
        sigs.append({"date":m["date"],"score":sc,"odds":best,"won":m["result"]=="D",
                     "season":m.get("season",""),"league":m.get("league_code","")})
    return pd.DataFrame(sigs)


def simulate_combos(bets, n_legs, max_per_day=3, stake=10):
    """Simule des combines reels."""
    bets = bets.copy()
    bets["ds"] = bets["date"].dt.strftime("%Y-%m-%d")
    total_s, total_r, n = 0, 0, 0
    for _, day in bets.groupby("ds"):
        if len(day) < n_legs: continue
        idxs = day.index.tolist()
        if len(idxs) > 8: idxs = day.nlargest(8,"score").index.tolist()
        combos = list(combinations(idxs, n_legs))
        np.random.shuffle(combos)
        for combo in combos[:max_per_day]:
            legs = bets.loc[list(combo)]
            co = legs["odds"].prod()
            total_s += stake
            if legs["won"].all(): total_r += stake * co
            n += 1
    return (total_r - total_s) / total_s * 100 if total_s > 0 else 0, n


def perm_test_combos(bets, n_legs, n_perm=500, max_per_day=3, stake=10):
    """Permutation test pour les combines."""
    real_roi, n_slips = simulate_combos(bets, n_legs, max_per_day, stake)

    # Permutation : melanger les resultats (won) entre les matchs
    rand_rois = []
    for _ in range(n_perm):
        shuffled = bets.copy()
        shuffled["won"] = np.random.permutation(shuffled["won"].values)
        r, _ = simulate_combos(shuffled, n_legs, max_per_day, stake)
        rand_rois.append(r)

    p = np.mean(np.array(rand_rois) >= real_roi)
    return {"real": real_roi, "n": n_slips, "rand_mean": np.mean(rand_rois),
            "rand_std": np.std(rand_rois), "p": p}


def main():
    print("=" * 60)
    print("VALIDATION RIGOUREUSE DES BET SLIPS")
    print("=" * 60)

    sig = load_and_score()
    print(f"   {len(sig)} signaux charges")

    for thresh in [4, 5]:
        sub = sig[sig["score"] >= thresh]
        print(f"\n{'='*60}")
        print(f"SCORE >= {thresh} ({len(sub)} paris)")
        print(f"{'='*60}")

        print(f"\n   {'Type':<12} {'Slips':>6} {'ROI reel':>10} {'ROI rand':>10} {'p-value':>9} {'Verdict':>12}")
        print(f"   {'-'*62}")

        # Simples
        roi_s = sub.apply(lambda b: 10*(b["odds"]-1) if b["won"] else -10, axis=1).sum()
        roi_s = roi_s / (len(sub)*10) * 100
        # Perm simples
        odds_arr = sub["odds"].values
        rand_s = []
        for _ in range(500):
            wins = np.random.permutation(sub["won"].values)
            p = sum(10*(o-1) if w else -10 for o,w in zip(odds_arr,wins))
            rand_s.append(p/(len(sub)*10)*100)
        p_s = np.mean(np.array(rand_s) >= roi_s)
        v = "SIGNAL" if p_s < 0.05 else ("Prometteur" if p_s < 0.15 else "-")
        print(f"   {'Simples':<12} {len(sub):>6} {roi_s:>+9.1f}% {np.mean(rand_s):>+9.1f}% {p_s:>8.4f} {v:>12}")

        # Doubles
        r = perm_test_combos(sub, 2, n_perm=300)
        v = "SIGNAL" if r["p"] < 0.05 else ("Prometteur" if r["p"] < 0.15 else "-")
        print(f"   {'Doubles':<12} {r['n']:>6} {r['real']:>+9.1f}% {r['rand_mean']:>+9.1f}% {r['p']:>8.4f} {v:>12}")

        # Triples
        r = perm_test_combos(sub, 3, n_perm=300, max_per_day=2)
        v = "SIGNAL" if r["p"] < 0.05 else ("Prometteur" if r["p"] < 0.15 else "-")
        print(f"   {'Triples':<12} {r['n']:>6} {r['real']:>+9.1f}% {r['rand_mean']:>+9.1f}% {r['p']:>8.4f} {v:>12}")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print("   p < 0.05 = le ROI des combines est SIGNIFICATIF")
    print("   p > 0.05 = le ROI vient de la variance, pas du signal")
    print("   Si simples sont significatifs ET combines aussi,")
    print("   les combines AMPLIFIENT un vrai edge.")


if __name__ == "__main__":
    main()
