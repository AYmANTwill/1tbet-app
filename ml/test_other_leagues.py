"""
test_other_leagues.py - Test de l'ensemble sur 2e divisions et ligues mineures

HYPOTHESE : Les marches moins suivis (2e divisions, petites ligues) sont
moins efficients donc plus exploitables.

LIGUES TESTEES :
- Championship (E1) : 2e division anglaise
- League One (E2) : 3e division anglaise
- Bundesliga 2 (D2)
- Serie B (I2)
- La Liga 2 (SP2)
- Ligue 2 (F2)
- Eredivisie (N1) : Pays-Bas
- Liga Portugal (P1)
- Belgian Pro League (B1)
- Scottish Premiership (SC0)

Usage : python ml/test_other_leagues.py
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


SEASONS = ["2021", "2122", "2223", "2324", "2425"]
LEAGUES = {
    "E1": "Championship",
    "E2": "League One",
    "D2": "Bundesliga 2",
    "I2": "Serie B",
    "SP2": "La Liga 2",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "P1": "Liga Portugal",
    "B1": "Belgian Pro",
    "SC0": "Scottish Prem",
}

BASE = "https://www.football-data.co.uk/mmz4281"
RAW_DIR = Path("data/raw_other")


def download():
    """Telecharge les CSV des autres ligues."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Telechargement des autres ligues...")
    for code, name in LEAGUES.items():
        for season in SEASONS:
            fname = RAW_DIR / f"{code}_{season}.csv"
            if fname.exists():
                continue
            url = f"{BASE}/{season}/{code}.csv"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    fname.write_bytes(r.content)
                    print(f"   OK {code}_{season} ({len(r.content)} bytes)")
                else:
                    print(f"   SKIP {code}_{season} (HTTP {r.status_code})")
            except Exception as e:
                print(f"   ERR {code}_{season}: {e}")


def load_data():
    """Charge tous les CSV avec ratings Elo."""
    dfs = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        try:
            d = pd.read_csv(f, encoding="latin-1")
            parts = f.stem.split("_")
            d["league_code"] = parts[0]
            d["season"] = "_".join(parts[1:])
            dfs.append(d)
        except:
            pass
    if not dfs:
        print("Aucune donnee, lance download d'abord")
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"HomeTeam":"home_team","AwayTeam":"away_team",
                             "FTHG":"home_goals","FTAG":"away_goals","FTR":"result"})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date","result"]).sort_values("date").reset_index(drop=True)
    for c in [col for col in df.columns if any(col.startswith(p) for p in ["B365","BW","IW","PS","WH","VC"])]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Elo par ligue (chaque ligue a ses propres ratings)
    df["elo_close"] = 0.5
    for lg in df["league_code"].unique():
        mask = df["league_code"] == lg
        sub = df[mask].copy()
        elo = {}
        K, HA = 25, 80
        ec = []
        for _, r in sub.iterrows():
            ht, at = r["home_team"], r["away_team"]
            ec.append(1/(1+abs(elo.get(ht,1500)-elo.get(at,1500))/100))
            hr = elo.get(ht,1500) + HA
            ar = elo.get(at,1500)
            e = 1/(1+10**((ar-hr)/400))
            s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
            elo[ht] = elo.get(ht,1500) + K*(s-e)
            elo[at] = elo.get(at,1500) + K*((1-s)-(1-e))
        df.loc[mask, "elo_close"] = ec
    return df


def compute_signals(df):
    """Score 0-6 pour chaque pari Draw."""
    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    sig = []
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
        s = 0
        if edge >= 0.03: s += 1
        if m["elo_close"] >= 0.5: s += 1
        med = np.median(vals)
        agree = sum(1 for o in vals if abs(o-med)<0.1*med)/len(vals)
        if agree >= 0.6: s += 1
        if not best_bk.startswith("PS"): s += 1
        if best > 3.0: s += 1
        if best_bk in ["B365D","BWD","WHD","VCD"]: s += 1
        sig.append({"score":s,"odds":best,"won":m["result"]=="D",
                    "season":m.get("season",""),"league":m.get("league_code","")})
    return pd.DataFrame(sig)


def backtest(bets, stake=10):
    if bets.empty: return {"n":0,"roi":0,"wr":0}
    profits = bets.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
    return {"n":len(bets), "roi":profits.sum()/(len(bets)*stake)*100,
            "wr":bets["won"].mean()}


def perm_test(bets, n=1000):
    real = backtest(bets)["roi"]
    odds = bets["odds"].values
    rand = []
    for _ in range(n):
        wins = np.random.random(len(odds)) < (1/odds)
        p = sum(10*(o-1) if w else -10 for o,w in zip(odds,wins))
        rand.append(p/(len(odds)*10)*100)
    return {"real":real,"rand_mean":np.mean(rand),"p":np.mean(np.array(rand)>=real)}


def main():
    print("=" * 60)
    print("TEST DE GENERALISABILITE SUR AUTRES LIGUES")
    print("=" * 60)

    if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.csv")):
        download()

    df = load_data()
    if df is None: return
    print(f"\n   {len(df)} matchs charges")
    print(f"   Ligues: {df['league_code'].unique().tolist()}")

    sig = compute_signals(df)
    print(f"   {len(sig)} signaux Draw extraits")

    # Test global
    print("\n" + "=" * 60)
    print("RESULTATS GLOBAUX (toutes ligues combinees)")
    print("=" * 60)
    for thresh in [3, 4, 5]:
        sub = sig[sig["score"] >= thresh]
        if len(sub) >= 50:
            r = backtest(sub)
            pt = perm_test(sub)
            v = "VRAI SIGNAL" if pt["p"] < 0.05 else ("PROMETTEUR" if pt["p"] < 0.20 else "BRUIT")
            print(f"\n   Score >= {thresh}: {r['n']} paris, ROI {r['roi']:+.1f}%")
            print(f"     Win rate: {r['wr']:.1%}")
            print(f"     Permutation p-value: {pt['p']:.4f} -> {v}")

    # Test par ligue
    print("\n" + "=" * 60)
    print("RESULTATS PAR LIGUE (Score >= 4)")
    print("=" * 60)
    print(f"\n   {'Ligue':<20} {'Paris':>6} {'Win%':>7} {'ROI':>8} {'p-value':>9} {'Verdict':>12}")
    print(f"   {'-'*65}")
    for lg in sorted(sig["league"].unique()):
        sub = sig[(sig["league"]==lg) & (sig["score"]>=4)]
        if len(sub) >= 30:
            r = backtest(sub)
            pt = perm_test(sub, n=500)
            v = "SIGNIFICATIF" if pt["p"] < 0.05 else ("Prometteur" if pt["p"] < 0.20 else "-")
            name = LEAGUES.get(lg, lg)
            print(f"   {name:<20} {r['n']:>6} {r['wr']:>6.1%} {r['roi']:>+7.1f}% {pt['p']:>8.4f} {v:>12}")

    # Test combiné Score >= 5 par ligue
    print("\n" + "=" * 60)
    print("RESULTATS PAR LIGUE (Score >= 5, plus selectif)")
    print("=" * 60)
    print(f"\n   {'Ligue':<20} {'Paris':>6} {'Win%':>7} {'ROI':>8} {'p-value':>9}")
    print(f"   {'-'*55}")
    for lg in sorted(sig["league"].unique()):
        sub = sig[(sig["league"]==lg) & (sig["score"]>=5)]
        if len(sub) >= 20:
            r = backtest(sub)
            pt = perm_test(sub, n=500)
            name = LEAGUES.get(lg, lg)
            m = " $$$" if pt["p"] < 0.05 and r["roi"] > 0 else ""
            print(f"   {name:<20} {r['n']:>6} {r['wr']:>6.1%} {r['roi']:>+7.1f}% {pt['p']:>8.4f}{m}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("   Si plusieurs ligues montrent p < 0.05, le signal est generalisable.")
    print("   Si seulement 1-2 ligues, c'est specifique a ces marches.")


if __name__ == "__main__":
    main()
