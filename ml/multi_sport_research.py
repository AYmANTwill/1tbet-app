"""
multi_sport_research.py - Extension multi-sport + feature engineering avance

SPORTS TESTES :
1. Football (notre base - revisiter les features)
2. NBA (basketball-reference via The Odds API)
3. Tennis (tennis-data.co.uk - structure similaire a football-data)
4. UFC/MMA (The Odds API)

APPROCHE :
La methode Kaunitz est sport-agnostique : on cherche les desaccords
entre bookmakers, pas les stats du sport. On peut l'appliquer partout.

NOUVEAU FEATURE ENGINEERING :
- Head-to-head historique
- Momentum (serie de victoires/defaites)
- Fatigue (matchs joues recemment)
- Home/Away split avance

Usage : python ml/multi_sport_research.py
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PARTIE 1 : TENNIS (tennis-data.co.uk)
# ============================================================

def download_tennis():
    """Telecharge les donnees tennis."""
    out = Path("data/raw_tennis")
    out.mkdir(parents=True, exist_ok=True)

    # ATP tours principaux
    base = "http://www.tennis-data.co.uk"
    years = ["2021", "2022", "2023", "2024", "2025"]

    for year in years:
        fname = out / f"atp_{year}.csv"
        if fname.exists(): continue
        url = f"{base}/{year}/{year}.xlsx"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                fname_xlsx = out / f"atp_{year}.xlsx"
                fname_xlsx.write_bytes(r.content)
                # Convertir xlsx en csv
                try:
                    df = pd.read_excel(fname_xlsx)
                    df.to_csv(fname, index=False)
                    print(f"   OK tennis {year} ({len(df)} matchs)")
                except:
                    print(f"   SKIP tennis {year} (excel parse error)")
            else:
                # Essayer le CSV direct
                url2 = f"{base}/{year}/{year}.csv"
                r2 = requests.get(url2, timeout=30)
                if r2.status_code == 200 and len(r2.content) > 5000:
                    fname.write_bytes(r2.content)
                    print(f"   OK tennis {year} (CSV)")
                else:
                    print(f"   SKIP tennis {year} (HTTP {r.status_code})")
        except Exception as e:
            print(f"   ERR tennis {year}: {e}")


def analyze_tennis():
    """Analyse Kaunitz sur le tennis."""
    print("\n" + "=" * 60)
    print("TENNIS : ANALYSE KAUNITZ")
    print("=" * 60)

    raw = Path("data/raw_tennis")
    if not raw.exists() or not list(raw.glob("*.csv")):
        print("   Pas de donnees tennis. Telechargement...")
        download_tennis()

    dfs = []
    for f in sorted(raw.glob("*.csv")):
        try:
            d = pd.read_csv(f, encoding="latin-1")
            d["source"] = f.stem
            dfs.append(d)
        except: pass

    if not dfs:
        print("   Aucune donnee tennis chargee")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"   {len(df)} matchs tennis charges")

    # Identifier les colonnes de cotes
    # Tennis-data a B365W, B365L, PSW, PSL etc (Winner/Loser)
    bk_w = [c for c in df.columns if any(c.startswith(p) for p in ["B365W","BWW","PSW","EXW"]) or c in ["B365W","PSW"]]
    bk_l = [c for c in df.columns if any(c.startswith(p) for p in ["B365L","BWL","PSL","EXL"]) or c in ["B365L","PSL"]]

    if not bk_w:
        # Chercher d'autres formats
        print(f"   Colonnes disponibles: {[c for c in df.columns if 'B365' in c or 'PS' in c or 'BW' in c]}")
        # Essayer le format standard
        bk_w = [c for c in ["B365W", "PSW", "EXW", "LBW", "SJW"] if c in df.columns]
        bk_l = [c for c in ["B365L", "PSL", "EXL", "LBL", "SJL"] if c in df.columns]

    for c in bk_w + bk_l:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"   Bookmakers W: {bk_w}")
    print(f"   Bookmakers L: {bk_l}")

    if len(bk_w) < 2:
        print("   Pas assez de bookmakers pour Kaunitz")
        return

    # Appliquer Kaunitz sur le tennis
    # Le tennis n'a que 2 outcomes (pas de Draw)
    # On cherche les mispricings sur le favori ET l'outsider
    stake = 10
    bets = []

    for _, m in df.iterrows():
        for outcome, bk_cols, won_col in [
            ("W", bk_w, True),   # Parier sur le Winner
            ("L", bk_l, False),  # Parier sur le Loser (upset)
        ]:
            odds = {}
            for c in bk_cols:
                v = m.get(c)
                if pd.notna(v) and v > 1.0:
                    odds[c] = v

            if len(odds) < 2: continue
            vals = list(odds.values())
            consensus = np.mean([1/o for o in vals])
            best_bk = max(odds, key=odds.get)
            best = odds[best_bk]
            edge = consensus - 1/best

            if edge < 0.02: continue

            # Score simple
            score = 0
            if edge >= 0.03: score += 1
            if not best_bk.startswith("PS"): score += 1
            if best > 2.0: score += 1
            med = np.median(vals)
            agree = sum(1 for o in vals if abs(o-med)<0.1*med)/len(vals)
            if agree >= 0.5: score += 1

            bets.append({
                "type": outcome, "odds": best, "edge": edge,
                "won": won_col, "score": score,
                "source": m.get("source", ""),
            })

    if not bets:
        print("   Aucun signal Kaunitz trouve")
        return

    bdf = pd.DataFrame(bets)
    print(f"   {len(bdf)} signaux extraits")

    for thresh in [2, 3]:
        sub = bdf[bdf["score"] >= thresh]
        if len(sub) >= 20:
            profit = sub.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
            roi = profit.sum() / (len(sub)*stake) * 100
            wr = sub["won"].mean()
            print(f"   Score >= {thresh}: {len(sub)} paris, Win {wr:.1%}, ROI {roi:+.1f}%")

            # Par type
            for t in ["W", "L"]:
                ts = sub[sub["type"]==t]
                if len(ts) >= 10:
                    p = ts.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
                    print(f"     {t}: {len(ts)} paris, ROI {p.sum()/(len(ts)*stake)*100:+.1f}%")


# ============================================================
# PARTIE 2 : NBA (The Odds API historique)
# ============================================================

def analyze_nba_live():
    """Analyse Kaunitz sur le NBA avec cotes live."""
    import os
    key = os.getenv("ODDS_API_KEY", "")

    print("\n" + "=" * 60)
    print("NBA : ANALYSE KAUNITZ (cotes live)")
    print("=" * 60)

    if not key:
        print("   Pas de cle API. Set ODDS_API_KEY pour le NBA live.")
        print("   Analyse demo...")
        return

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    resp = requests.get(url, params={
        "apiKey": key, "regions": "us,uk,eu",
        "oddsFormat": "decimal", "markets": "h2h",
    }, timeout=30)

    if resp.status_code != 200:
        print(f"   API error {resp.status_code}")
        return

    events = resp.json()
    print(f"   {len(events)} matchs NBA recuperes")

    picks = []
    for event in events:
        home = event["home_team"]
        away = event["away_team"]

        # Extraire cotes
        odds_h, odds_a = {}, {}
        for bk in event.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        if o["name"] == home:
                            odds_h[bk["title"]] = o["price"]
                        elif o["name"] == away:
                            odds_a[bk["title"]] = o["price"]

        # Analyser les deux cotes
        for label, odds_dict, team in [("H", odds_h, home), ("A", odds_a, away)]:
            if len(odds_dict) < 3: continue
            vals = list(odds_dict.values())
            consensus = np.mean([1/o for o in vals])
            best_bk = max(odds_dict, key=odds_dict.get)
            best = odds_dict[best_bk]
            edge = consensus - 1/best

            if edge < 0.02: continue

            score = 0
            if edge >= 0.03: score += 1
            if not any(best_bk.startswith(s) for s in ["Pinnacle","BetOnline"]): score += 1
            if best > 2.0: score += 1
            med = np.median(vals)
            if sum(1 for o in vals if abs(o-med)<0.1*med)/len(vals) >= 0.5: score += 1

            if score >= 3:
                picks.append({
                    "match": f"{home} vs {away}",
                    "team": team, "odds": best, "edge": edge,
                    "score": score, "bookmaker": best_bk,
                    "time": event.get("commence_time",""),
                })

    if picks:
        print(f"   {len(picks)} signaux NBA detectes :")
        for p in sorted(picks, key=lambda x: x["score"], reverse=True):
            print(f"     [{p['score']}/4] {p['match']} -> {p['team']} "
                  f"@ {p['odds']:.2f} ({p['bookmaker']}) edge={p['edge']:.1%}")
    else:
        print("   Aucun signal NBA pour le moment")


# ============================================================
# PARTIE 3 : UFC/MMA (The Odds API)
# ============================================================

def analyze_mma_live():
    """Analyse Kaunitz sur le MMA."""
    import os
    key = os.getenv("ODDS_API_KEY", "")

    print("\n" + "=" * 60)
    print("UFC/MMA : ANALYSE KAUNITZ (cotes live)")
    print("=" * 60)

    if not key:
        print("   Pas de cle API.")
        return

    url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
    resp = requests.get(url, params={
        "apiKey": key, "regions": "us,uk,eu",
        "oddsFormat": "decimal", "markets": "h2h",
    }, timeout=30)

    if resp.status_code != 200:
        print(f"   API error {resp.status_code}")
        return

    events = resp.json()
    print(f"   {len(events)} combats MMA recuperes")

    picks = []
    for event in events:
        h, a = event["home_team"], event["away_team"]
        odds_h, odds_a = {}, {}
        for bk in event.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h":
                    for o in mkt["outcomes"]:
                        if o["name"] == h: odds_h[bk["title"]] = o["price"]
                        elif o["name"] == a: odds_a[bk["title"]] = o["price"]

        for label, od, team in [("H",odds_h,h),("A",odds_a,a)]:
            if len(od) < 3: continue
            vals = list(od.values())
            cons = np.mean([1/o for o in vals])
            bb = max(od, key=od.get)
            bo = od[bb]
            edge = cons - 1/bo
            if edge < 0.03: continue

            score = 0
            if edge >= 0.04: score += 1
            if not any(bb.startswith(s) for s in ["Pinnacle","BetOnline"]): score += 1
            if bo > 2.0: score += 1
            med = np.median(vals)
            if sum(1 for o in vals if abs(o-med)<0.1*med)/len(vals) >= 0.5: score += 1

            if score >= 2:
                picks.append({"match":f"{h} vs {a}","team":team,"odds":bo,
                              "edge":edge,"score":score,"bk":bb,
                              "time":event.get("commence_time","")})

    if picks:
        print(f"   {len(picks)} signaux MMA :")
        for p in sorted(picks, key=lambda x: x["score"], reverse=True):
            print(f"     [{p['score']}/4] {p['match']} -> {p['team']} "
                  f"@ {p['odds']:.2f} ({p['bk']}) edge={p['edge']:.1%}")
    else:
        print("   Aucun signal MMA")


# ============================================================
# PARTIE 4 : FEATURE ENGINEERING V2 (Football)
# ============================================================

def advanced_feature_engineering():
    """
    Nouvelles features pour le football.

    CE QUI MANQUAIT :
    1. Momentum (serie de V/D)
    2. Head-to-head historique
    3. Fatigue (matchs en 7 jours)
    4. Home/Away split (certaines equipes sont meilleures dehors)
    5. Position au classement estimee
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING V2 (Football)")
    print("=" * 60)

    dfs = []
    for f in sorted(Path("data/raw").glob("*.csv")):
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

    print(f"   {len(df)} matchs football")

    # 1. MOMENTUM : serie de resultats sans defaite
    print("   Calcul momentum...")
    momentum_h, momentum_a = [], []
    team_streaks = {}

    for _, r in df.iterrows():
        ht, at = r["home_team"], r["away_team"]
        momentum_h.append(team_streaks.get(ht, 0))
        momentum_a.append(team_streaks.get(at, 0))

        if r["result"] == "H":
            team_streaks[ht] = team_streaks.get(ht, 0) + 1
            team_streaks[at] = min(0, team_streaks.get(at, 0)) - 1
        elif r["result"] == "A":
            team_streaks[at] = team_streaks.get(at, 0) + 1
            team_streaks[ht] = min(0, team_streaks.get(ht, 0)) - 1
        else:
            team_streaks[ht] = 0
            team_streaks[at] = 0

    df["momentum_home"] = momentum_h
    df["momentum_away"] = momentum_a
    df["momentum_diff"] = df["momentum_home"] - df["momentum_away"]

    # 2. FATIGUE : matchs dans les 7 derniers jours
    print("   Calcul fatigue...")
    # Simplifie : compter les matchs par equipe dans les 7 jours precedents
    df["fatigue_home"] = 0
    df["fatigue_away"] = 0
    # Approximation rapide par groupby
    for team in df["home_team"].unique():
        mask_h = df["home_team"] == team
        mask_a = df["away_team"] == team
        team_dates = df.loc[mask_h | mask_a, "date"].sort_values()
        # Pour chaque match, compter les matchs dans les 7j precedents
        for idx in df[mask_h].index:
            d = df.loc[idx, "date"]
            recent = ((team_dates < d) & (team_dates >= d - pd.Timedelta(days=7))).sum()
            df.loc[idx, "fatigue_home"] = recent
        for idx in df[mask_a].index:
            d = df.loc[idx, "date"]
            recent = ((team_dates < d) & (team_dates >= d - pd.Timedelta(days=7))).sum()
            df.loc[idx, "fatigue_away"] = recent

    # 3. HEAD-TO-HEAD : resultat du dernier confrontation directe
    print("   Calcul H2H...")
    h2h_draws = []
    h2h_cache = {}

    for _, r in df.iterrows():
        key = tuple(sorted([r["home_team"], r["away_team"]]))
        h2h_draws.append(h2h_cache.get(key, 0.25))  # Default 25%

        # Mettre a jour le cache
        if key not in h2h_cache:
            h2h_cache[key] = 1.0 if r["result"] == "D" else 0.0
        else:
            # Moyenne mobile
            alpha = 0.3
            is_draw = 1.0 if r["result"] == "D" else 0.0
            h2h_cache[key] = alpha * is_draw + (1-alpha) * h2h_cache[key]

    df["h2h_draw_rate"] = h2h_draws

    # 4. POSITION AU CLASSEMENT (estimee par points cumules)
    print("   Calcul classement...")
    points = {}
    pos_h, pos_a = [], []

    for _, r in df.iterrows():
        ht, at = r["home_team"], r["away_team"]
        pos_h.append(points.get(ht, 0))
        pos_a.append(points.get(at, 0))

        if r["result"] == "H":
            points[ht] = points.get(ht, 0) + 3
        elif r["result"] == "D":
            points[ht] = points.get(ht, 0) + 1
            points[at] = points.get(at, 0) + 1
        else:
            points[at] = points.get(at, 0) + 3

    df["points_home"] = pos_h
    df["points_away"] = pos_a
    df["points_diff"] = df["points_home"] - df["points_away"]

    # ANALYSER l'impact de ces features sur les draws
    print(f"\n   IMPACT DES NOUVELLES FEATURES SUR LE DRAW :")
    draws = df[df["result"] == "D"]
    non_draws = df[df["result"] != "D"]

    features_to_check = [
        ("momentum_diff", "Momentum equilibre"),
        ("fatigue_home", "Fatigue domicile"),
        ("fatigue_away", "Fatigue exterieur"),
        ("h2h_draw_rate", "H2H taux de draw"),
        ("points_diff", "Difference de points"),
    ]

    print(f"\n   {'Feature':<25} {'Draws':>10} {'Non-draws':>12} {'Signal?':>10}")
    print(f"   {'-'*60}")

    for feat, label in features_to_check:
        if feat in draws.columns:
            d_mean = draws[feat].mean()
            nd_mean = non_draws[feat].mean()
            diff = abs(d_mean - nd_mean)
            signal = "OUI" if diff > 0.1 * abs(nd_mean + 0.01) else "non"
            print(f"   {label:<25} {d_mean:>10.2f} {nd_mean:>12.2f} {signal:>10}")

    # Tester si les nouvelles features aident le Kaunitz
    print(f"\n   KAUNITZ + NOUVELLES FEATURES :")
    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    for c in bk_d:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filtrer par momentum equilibre (abs(diff) <= 2)
    balanced_momentum = df[abs(df["momentum_diff"]) <= 2]
    # Filtrer par H2H draw rate > 0.3
    h2h_favorable = df[df["h2h_draw_rate"] > 0.3]

    stake = 10
    for label, subset in [
        ("Baseline (tous)", df),
        ("Momentum equilibre", balanced_momentum),
        ("H2H favorable au draw", h2h_favorable),
        ("Momentum + H2H", df[(abs(df["momentum_diff"])<=2) & (df["h2h_draw_rate"]>0.3)]),
    ]:
        bets = []
        for _, m in subset.iterrows():
            odds = {c: m[c] for c in bk_d if pd.notna(m.get(c)) and m.get(c) > 1}
            if len(odds) < 3: continue
            vals = list(odds.values())
            cons = np.mean([1/o for o in vals])
            bb = max(odds, key=odds.get)
            bo = odds[bb]
            edge = cons - 1/bo
            if edge < 0.03: continue
            sc = 0
            if edge >= 0.05: sc += 1
            if not bb.startswith("PS"): sc += 1
            if bo > 3.0: sc += 1
            if bb in ["B365D","BWD","WHD","VCD"]: sc += 1
            if sc >= 3:
                won = m["result"] == "D"
                bets.append({"won":won,"odds":bo})

        if bets:
            bdf = pd.DataFrame(bets)
            p = bdf.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
            roi = p.sum() / (len(bdf)*stake) * 100
            print(f"   {label:<30} {len(bdf):>5} paris, ROI {roi:>+7.1f}%")


def main():
    import os
    print("=" * 60)
    print("MULTI-SPORT RESEARCH + FEATURE ENGINEERING V2")
    print("=" * 60)

    # Football features
    advanced_feature_engineering()

    # Tennis
    analyze_tennis()

    # NBA (si API key dispo)
    if os.getenv("ODDS_API_KEY"):
        analyze_nba_live()
        analyze_mma_live()
    else:
        print("\n   NBA/MMA : set ODDS_API_KEY pour analyser les cotes live")

    print("\n" + "=" * 60)
    print("RECHERCHE MULTI-SPORT TERMINEE")
    print("=" * 60)


if __name__ == "__main__":
    main()
