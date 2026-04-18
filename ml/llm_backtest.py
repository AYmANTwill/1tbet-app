"""
llm_backtest.py - Backtest du LLM local sur matchs historiques

Passe chaque signal Draw Score>=8 au LLM local (Ollama)
avec le contexte historique du match. Le LLM repond BET/SKIP.
On compare le ROI avec et sans filtre LLM.

IMPORTANT : ca prend ~2-3h sur GTX 1660 Ti pour ~1000 matchs.
Sauvegarde les resultats au fur et a mesure dans un CSV.

Usage :
    ollama serve   (dans un autre terminal si pas deja actif)
    python ml/llm_backtest.py
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
RESULTS_FILE = Path("data/llm_backtest_results.csv")

PROMPT_TEMPLATE = """You are a football betting analyst. Analyze this match for DRAW probability.

MATCH: {home} vs {away}
LEAGUE: {league}
SEASON: {season}

HOME TEAM CONTEXT:
- Recent form (goals scored last 5): {home_form:.1f}
- Recent defense (goals conceded last 5): {home_def:.1f}
- Elo rating: {home_elo:.0f}

AWAY TEAM CONTEXT:
- Recent form (goals scored last 5): {away_form:.1f}
- Recent defense (goals conceded last 5): {away_def:.1f}
- Elo rating: {away_elo:.0f}

BETTING CONTEXT:
- Draw odds: {draw_odds:.2f}
- Statistical edge detected: {edge:.1%}
- Bookmaker disagreement: {n_outliers} bookmaker(s) offering above consensus
- Match balance (Elo closeness): {elo_close:.2f} (1.0 = perfectly balanced)

Based on this context, should we bet on DRAW?

Answer ONLY with a JSON object:
{{"verdict": "BET" or "SKIP", "confidence": 0.0 to 1.0, "reason": "one short sentence"}}"""


def load_and_prep():
    """Charge les donnees avec contexte complet."""
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
    elo_h, elo_a, ec = [], [], []
    for _, r in df.iterrows():
        lg = r["league_code"]
        if lg not in elos: elos[lg] = {}
        elo = elos[lg]
        ht, at = r["home_team"], r["away_team"]
        elo_h.append(elo.get(ht, 1500))
        elo_a.append(elo.get(at, 1500))
        ec.append(1/(1+abs(elo.get(ht,1500)-elo.get(at,1500))/100))
        hr = elo.get(ht,1500) + HA
        ar = elo.get(at,1500)
        e = 1/(1+10**((ar-hr)/400))
        s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht] = elo.get(ht,1500) + K*(s-e)
        elo[at] = elo.get(at,1500) + K*((1-s)-(1-e))
    df["elo_home"] = elo_h
    df["elo_away"] = elo_a
    df["elo_close"] = ec

    # Forme
    df["home_form"] = df.groupby("home_team")["home_goals"].transform(
        lambda x: x.rolling(5,min_periods=1).mean())
    df["away_form"] = df.groupby("away_team")["away_goals"].transform(
        lambda x: x.rolling(5,min_periods=1).mean())
    df["home_form"] = df.groupby("home_team")["home_form"].shift(1)
    df["away_form"] = df.groupby("away_team")["away_form"].shift(1)
    df["home_def"] = df.groupby("home_team")["away_goals"].transform(
        lambda x: x.rolling(5,min_periods=1).mean())
    df["away_def"] = df.groupby("away_team")["home_goals"].transform(
        lambda x: x.rolling(5,min_periods=1).mean())
    df["home_def"] = df.groupby("home_team")["home_def"].shift(1)
    df["away_def"] = df.groupby("away_team")["away_def"].shift(1)

    return df


def compute_ensemble_score(row, bk_d):
    """Calcule le score ensemble pour un match."""
    odds = {c: row[c] for c in bk_d if pd.notna(row.get(c)) and row.get(c) > 1}
    if len(odds) < 3: return None, None, None, None
    vals = list(odds.values())
    consensus = np.mean([1/o for o in vals])
    best_bk = max(odds, key=odds.get)
    best = odds[best_bk]
    edge = consensus - 1/best
    if edge < 0.01: return None, None, None, None

    s = 0
    if edge >= 0.03: s += 1
    if row.get("elo_close",0) >= 0.5: s += 1
    med = np.median(vals)
    agree = sum(1 for o in vals if abs(o-med)<0.1*med)/len(vals)
    if agree >= 0.6: s += 1
    if not best_bk.startswith("PS"): s += 1
    if best > 3.0: s += 1
    if best_bk in ["B365D","BWD","WHD","VCD"]: s += 1

    n_outliers = sum(1 for o in vals if o > (1/consensus)*1.03)
    return s, best, edge, n_outliers


def ask_llm(prompt, timeout=30):
    """Appel Ollama local."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150}
        }, timeout=timeout)
        if resp.status_code == 200:
            text = resp.json().get("response", "")
            # Extraire le JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        return None
    except:
        return None


def main():
    print("=" * 60)
    print("LLM BACKTEST : OLLAMA LOCAL SUR MATCHS HISTORIQUES")
    print("=" * 60)

    df = load_and_prep()
    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    print(f"   {len(df)} matchs charges")

    # Filtrer les matchs avec score >= 8 (notre meilleur seuil)
    candidates = []
    for idx, row in df.iterrows():
        score, best_odds, edge, n_out = compute_ensemble_score(row, bk_d)
        if score is not None and score >= 4:
            candidates.append({
                "idx": idx, "score": score, "odds": best_odds,
                "edge": edge, "n_outliers": n_out,
                "home": row["home_team"], "away": row["away_team"],
                "league": row.get("league_code",""), "season": row.get("season",""),
                "home_form": row.get("home_form", 1.3),
                "away_form": row.get("away_form", 1.1),
                "home_def": row.get("home_def", 1.2),
                "away_def": row.get("away_def", 1.3),
                "home_elo": row.get("elo_home", 1500),
                "away_elo": row.get("elo_away", 1500),
                "elo_close": row.get("elo_close", 0.5),
                "draw_odds": best_odds,
                "won": row["result"] == "D",
            })

    print(f"   {len(candidates)} paris Score >= 8 a analyser")

    # Charger resultats deja calcules
    done = set()
    if RESULTS_FILE.exists():
        prev = pd.read_csv(RESULTS_FILE)
        done = set(prev["idx"].values)
        print(f"   {len(done)} deja analyses (reprise)")

    # Analyser chaque match
    results = []
    if RESULTS_FILE.exists():
        results = pd.read_csv(RESULTS_FILE).to_dict("records")

    total = len(candidates)
    skipped = 0
    t0 = time.time()

    print(f"\n   Lancement du LLM sur {total - len(done)} matchs...")
    print(f"   Estimation : ~{(total - len(done)) * 3 / 60:.0f} minutes")
    print(f"   {'='*50}")

    for i, c in enumerate(candidates):
        if c["idx"] in done:
            continue

        prompt = PROMPT_TEMPLATE.format(
            home=c["home"], away=c["away"], league=c["league"],
            season=c["season"], home_form=c.get("home_form",1.3),
            home_def=c.get("home_def",1.2), home_elo=c["home_elo"],
            away_form=c.get("away_form",1.1), away_def=c.get("away_def",1.3),
            away_elo=c["away_elo"], draw_odds=c["draw_odds"],
            edge=c["edge"], n_outliers=c.get("n_outliers",1),
            elo_close=c["elo_close"],
        )

        llm = ask_llm(prompt)

        if llm:
            verdict = llm.get("verdict", "SKIP")
            conf = llm.get("confidence", 0.5)
            reason = llm.get("reason", "")
        else:
            verdict = "UNKNOWN"
            conf = 0.5
            reason = "LLM parse error"

        result = {
            "idx": c["idx"], "home": c["home"], "away": c["away"],
            "league": c["league"], "season": c["season"],
            "score": c["score"], "odds": c["odds"], "edge": c["edge"],
            "won": c["won"], "llm_verdict": verdict,
            "llm_confidence": conf, "llm_reason": reason,
        }
        results.append(result)
        done.add(c["idx"])

        # Sauvegarder toutes les 10 iterations
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

        # Progress
        elapsed = time.time() - t0
        done_count = len(done)
        remaining = total - done_count
        speed = (i + 1) / max(elapsed, 1)
        eta = remaining / max(speed, 0.01) / 60

        if (i + 1) % 25 == 0:
            print(f"   [{done_count}/{total}] {elapsed/60:.1f}min elapsed, "
                  f"~{eta:.0f}min remaining | Last: {verdict} {conf:.0%} {c['home']} vs {c['away']}")

    # Sauvegarder final
    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
    print(f"\n   Resultats sauvegardes : {RESULTS_FILE}")

    # =========================================
    # ANALYSE DES RESULTATS
    # =========================================
    rdf = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("RESULTATS DU BACKTEST LLM")
    print(f"{'='*60}")

    print(f"\n   Total analyse : {len(rdf)}")
    print(f"   LLM BET  : {(rdf['llm_verdict']=='BET').sum()}")
    print(f"   LLM SKIP : {(rdf['llm_verdict']=='SKIP').sum()}")

    # ROI sans LLM (baseline)
    stake = 10
    rdf["profit"] = rdf.apply(lambda r: stake*(r["odds"]-1) if r["won"] else -stake, axis=1)
    roi_all = rdf["profit"].sum() / (len(rdf) * stake) * 100
    wr_all = rdf["won"].mean()

    # ROI avec filtre LLM
    bet_only = rdf[rdf["llm_verdict"] == "BET"]
    if len(bet_only) > 0:
        roi_bet = bet_only["profit"].sum() / (len(bet_only) * stake) * 100
        wr_bet = bet_only["won"].mean()
    else:
        roi_bet, wr_bet = 0, 0

    skip_only = rdf[rdf["llm_verdict"] == "SKIP"]
    if len(skip_only) > 0:
        roi_skip = skip_only["profit"].sum() / (len(skip_only) * stake) * 100
    else:
        roi_skip = 0

    print(f"\n   {'Strategie':<25} {'Paris':>6} {'Win%':>7} {'ROI':>8}")
    print(f"   {'-'*48}")
    print(f"   {'Sans LLM (baseline)':<25} {len(rdf):>6} {wr_all:>6.1%} {roi_all:>+7.1f}%")
    print(f"   {'LLM = BET seulement':<25} {len(bet_only):>6} {wr_bet:>6.1%} {roi_bet:>+7.1f}%")
    print(f"   {'LLM = SKIP (evites)':<25} {len(skip_only):>6} {'':>7} {roi_skip:>+7.1f}%")

    improvement = roi_bet - roi_all
    print(f"\n   Amelioration LLM : {improvement:+.1f}%")
    if improvement > 0:
        print(f"   VERDICT : Le LLM AMELIORE la strategie de {improvement:.1f} points de ROI")
    else:
        print(f"   VERDICT : Le LLM N'AMELIORE PAS la strategie")

    # Par confiance LLM
    print(f"\n   ROI par confiance LLM :")
    for lo, hi, label in [(0.0, 0.4, "Faible"), (0.4, 0.6, "Moyen"),
                           (0.6, 0.8, "Fort"), (0.8, 1.01, "Tres fort")]:
        sub = rdf[(rdf["llm_confidence"] >= lo) & (rdf["llm_confidence"] < hi)]
        if len(sub) >= 10:
            r = sub["profit"].sum() / (len(sub) * stake) * 100
            print(f"   {label:<12} (conf {lo:.0%}-{hi:.0%}): {len(sub)} paris, ROI {r:+.1f}%")

    # Par saison
    print(f"\n   Walk-forward LLM BET :")
    for s in sorted(bet_only["season"].unique()):
        sub = bet_only[bet_only["season"] == s]
        if len(sub) >= 5:
            r = sub["profit"].sum() / (len(sub) * stake) * 100
            m = " $" if r > 0 else ""
            print(f"   {s}: {len(sub)} paris, ROI {r:+.1f}%{m}")


if __name__ == "__main__":
    main()
