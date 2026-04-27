"""
live_pipeline.py - Pipeline temps reel pour generer des bet slips

FLUX :
1. Recupere les cotes live (The Odds API)
2. Calcule le score ensemble pour chaque match
3. Filtre les meilleurs signaux Draw
4. Genere des bet slips (simples + doubles + triples)
5. Calcule le Kelly sizing pour chaque slip

Usage :
    set ODDS_API_KEY=ta-cle
    python ml/live_pipeline.py
    python ml/live_pipeline.py --sport soccer_spain_la_liga
"""

import os
import sys
import json
import requests
import numpy as np
from datetime import datetime
from pathlib import Path


ODDS_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE = "https://api.the-odds-api.com/v4"

# Ligues ou notre signal est valide (p < 0.05)
VALIDATED_LEAGUES = {
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_italy_serie_b": "Serie B",
    "soccer_spain_segunda_division": "La Liga 2",
    "soccer_netherlands_eredivisie": "Eredivisie",
}

# Bookmakers "soft" connus (plus susceptibles de mispricing)
SOFT_BOOKS = ["Bet365", "William Hill", "Betway", "Unibet", "888sport",
              "Betfair Sportsbook", "Coral", "Ladbrokes"]

# Bookmaker "sharp" (reference)
SHARP_BOOKS = ["Pinnacle"]


def fetch_odds(sport="soccer_epl"):
    """Recupere les cotes live."""
    if not ODDS_KEY:
        print("   [DEMO MODE] Pas de cle API")
        return _demo_data()

    url = f"{ODDS_BASE}/sports/{sport}/odds"
    resp = requests.get(url, params={
        "apiKey": ODDS_KEY, "regions": "uk,eu",
        "oddsFormat": "decimal", "markets": "h2h",
    }, timeout=30)

    if resp.status_code != 200:
        print(f"   API error {resp.status_code}")
        return _demo_data()

    return resp.json()


def analyze_match(event):
    """Analyse un match et calcule le score ensemble Draw."""
    home = event["home_team"]
    away = event["away_team"]

    # Extraire les cotes Draw de chaque bookmaker
    draw_odds = {}
    home_odds = {}
    away_odds = {}

    for bk in event.get("bookmakers", []):
        name = bk["title"]
        for market in bk.get("markets", []):
            if market["key"] == "h2h":
                for outcome in market["outcomes"]:
                    if outcome["name"] == "Draw":
                        draw_odds[name] = outcome["price"]
                    elif outcome["name"] == home:
                        home_odds[name] = outcome["price"]
                    elif outcome["name"] == away:
                        away_odds[name] = outcome["price"]

    if len(draw_odds) < 3:
        return None

    # Consensus
    vals = list(draw_odds.values())
    consensus = np.mean([1/o for o in vals])
    fair = 1 / consensus

    # Meilleure cote
    best_bk = max(draw_odds, key=draw_odds.get)
    best = draw_odds[best_bk]
    edge = consensus - 1/best

    if edge < 0.01:
        return None

    # =========================================
    # SCORE ENSEMBLE (0-6)
    # =========================================
    score = 0

    # 1. Edge significatif
    if edge >= 0.03: score += 1

    # 2. Match equilibre (cotes H et A proches)
    if home_odds and away_odds:
        avg_h = np.mean(list(home_odds.values()))
        avg_a = np.mean(list(away_odds.values()))
        ratio = max(avg_h, avg_a) / min(avg_h, avg_a)
        if ratio < 1.5: score += 1  # Proxy elo_close

    # 3. Fort accord (1 outlier vs le reste)
    med = np.median(vals)
    agree = sum(1 for o in vals if abs(o-med) < 0.1*med) / len(vals)
    if agree >= 0.6: score += 1

    # 4. Pinnacle PAS outlier
    is_pin_outlier = best_bk in SHARP_BOOKS
    if not is_pin_outlier: score += 1

    # 5. Cote > 3.0
    if best > 3.0: score += 1

    # 6. Soft bookmaker est l'outlier
    if any(best_bk.startswith(s.split()[0]) for s in SOFT_BOOKS): score += 1

    # Kelly fractionnel
    kelly = max(0, (consensus * best - 1) / (best - 1) * 0.25)
    kelly = min(kelly, 0.10)

    return {
        "home": home,
        "away": away,
        "commence": event.get("commence_time", ""),
        "score": score,
        "best_odds": best,
        "best_bookmaker": best_bk,
        "edge": round(edge, 4),
        "consensus_prob": round(consensus, 4),
        "n_bookmakers": len(draw_odds),
        "agreement": round(agree, 2),
        "kelly": round(kelly, 4),
        "all_draw_odds": draw_odds,
    }


def generate_betslips(picks, max_doubles=5, max_triples=3):
    """Genere des bet slips a partir des picks."""
    slips = []

    # SIMPLES
    for p in picks:
        slips.append({
            "type": "SIMPLE",
            "legs": [p],
            "combined_odds": p["best_odds"],
            "combined_edge": p["edge"],
            "kelly": p["kelly"],
        })

    # DOUBLES (cross-league de preference)
    if len(picks) >= 2:
        from itertools import combinations
        pairs = list(combinations(range(len(picks)), 2))
        # Trier par score combine decroissant
        pairs.sort(key=lambda pair: picks[pair[0]]["score"] + picks[pair[1]]["score"],
                   reverse=True)
        for i, j in pairs[:max_doubles]:
            combo_odds = picks[i]["best_odds"] * picks[j]["best_odds"]
            slips.append({
                "type": "DOUBLE",
                "legs": [picks[i], picks[j]],
                "combined_odds": round(combo_odds, 2),
                "combined_edge": picks[i]["edge"] + picks[j]["edge"],
                "kelly": min(picks[i]["kelly"], picks[j]["kelly"]) * 0.5,
            })

    # TRIPLES
    if len(picks) >= 3:
        from itertools import combinations
        triples = list(combinations(range(len(picks)), 3))
        triples.sort(key=lambda t: sum(picks[x]["score"] for x in t), reverse=True)
        for combo in triples[:max_triples]:
            combo_odds = np.prod([picks[x]["best_odds"] for x in combo])
            slips.append({
                "type": "TRIPLE",
                "legs": [picks[x] for x in combo],
                "combined_odds": round(combo_odds, 2),
                "combined_edge": sum(picks[x]["edge"] for x in combo),
                "kelly": min(picks[x]["kelly"] for x in combo) * 0.3,
            })

    return slips


def display_results(picks, slips, bankroll=1000):
    """Affiche les resultats."""
    print(f"\n{'='*60}")
    print(f"SIGNAUX DRAW DETECTES : {len(picks)}")
    print(f"{'='*60}")

    if not picks:
        print("   Aucun signal Draw detecte aujourd'hui.")
        return

    for p in sorted(picks, key=lambda x: x["score"], reverse=True):
        stars = "*" * p["score"]
        print(f"\n   [{stars:<6}] {p['home']} vs {p['away']}")
        print(f"           Score: {p['score']}/6 | Odds: {p['best_odds']:.2f} "
              f"({p['best_bookmaker']}) | Edge: {p['edge']:.1%}")
        print(f"           Kelly: {p['kelly']:.1%} du bankroll = {bankroll*p['kelly']:.0f} unites")

    # Bet slips
    print(f"\n{'='*60}")
    print(f"BET SLIPS RECOMMANDES")
    print(f"{'='*60}")

    for slip_type in ["SIMPLE", "DOUBLE", "TRIPLE"]:
        type_slips = [s for s in slips if s["type"] == slip_type]
        if not type_slips:
            continue

        print(f"\n   --- {slip_type}S ---")
        for s in type_slips[:5]:
            stake = bankroll * s["kelly"]
            potential = stake * s["combined_odds"]
            matches = " + ".join(f"{l['home']} vs {l['away']}" for l in s["legs"])
            print(f"   {matches}")
            print(f"     Odds: {s['combined_odds']:.1f} | Mise: {stake:.0f} | "
                  f"Gain potentiel: {potential:.0f}")

    # Resume
    n_simples = len([s for s in slips if s["type"] == "SIMPLE"])
    n_doubles = len([s for s in slips if s["type"] == "DOUBLE"])
    n_triples = len([s for s in slips if s["type"] == "TRIPLE"])
    total_stake = sum(bankroll * s["kelly"] for s in slips)

    print(f"\n{'='*60}")
    print(f"RESUME")
    print(f"{'='*60}")
    print(f"   Simples: {n_simples} | Doubles: {n_doubles} | Triples: {n_triples}")
    print(f"   Mise totale recommandee: {total_stake:.0f} ({total_stake/bankroll:.0%} du bankroll)")


def _demo_data():
    """Donnees demo."""
    return [
        {"home_team":"Everton","away_team":"Newcastle","commence_time":"2026-04-20T15:00:00Z",
         "bookmakers":[
            {"title":"Bet365","markets":[{"key":"h2h","outcomes":[
                {"name":"Everton","price":2.80},{"name":"Draw","price":3.50},{"name":"Newcastle","price":2.50}]}]},
            {"title":"Pinnacle","markets":[{"key":"h2h","outcomes":[
                {"name":"Everton","price":2.75},{"name":"Draw","price":3.30},{"name":"Newcastle","price":2.55}]}]},
            {"title":"William Hill","markets":[{"key":"h2h","outcomes":[
                {"name":"Everton","price":2.70},{"name":"Draw","price":3.60},{"name":"Newcastle","price":2.45}]}]},
            {"title":"Betway","markets":[{"key":"h2h","outcomes":[
                {"name":"Everton","price":2.85},{"name":"Draw","price":3.40},{"name":"Newcastle","price":2.50}]}]},
         ]},
        {"home_team":"Aston Villa","away_team":"Brighton","commence_time":"2026-04-20T15:00:00Z",
         "bookmakers":[
            {"title":"Bet365","markets":[{"key":"h2h","outcomes":[
                {"name":"Aston Villa","price":2.20},{"name":"Draw","price":3.40},{"name":"Brighton","price":3.30}]}]},
            {"title":"Pinnacle","markets":[{"key":"h2h","outcomes":[
                {"name":"Aston Villa","price":2.15},{"name":"Draw","price":3.35},{"name":"Brighton","price":3.25}]}]},
            {"title":"William Hill","markets":[{"key":"h2h","outcomes":[
                {"name":"Aston Villa","price":2.25},{"name":"Draw","price":3.50},{"name":"Brighton","price":3.20}]}]},
            {"title":"Betway","markets":[{"key":"h2h","outcomes":[
                {"name":"Aston Villa","price":2.20},{"name":"Draw","price":3.45},{"name":"Brighton","price":3.35}]}]},
         ]},
        {"home_team":"Man City","away_team":"Burnley","commence_time":"2026-04-20T17:30:00Z",
         "bookmakers":[
            {"title":"Bet365","markets":[{"key":"h2h","outcomes":[
                {"name":"Man City","price":1.15},{"name":"Draw","price":8.00},{"name":"Burnley","price":17.00}]}]},
            {"title":"Pinnacle","markets":[{"key":"h2h","outcomes":[
                {"name":"Man City","price":1.14},{"name":"Draw","price":8.50},{"name":"Burnley","price":18.00}]}]},
            {"title":"William Hill","markets":[{"key":"h2h","outcomes":[
                {"name":"Man City","price":1.16},{"name":"Draw","price":7.50},{"name":"Burnley","price":16.00}]}]},
            {"title":"Betway","markets":[{"key":"h2h","outcomes":[
                {"name":"Man City","price":1.15},{"name":"Draw","price":8.00},{"name":"Burnley","price":17.00}]}]},
         ]},
    ]


def main():
    sport = "soccer_epl"
    if "--sport" in sys.argv:
        idx = sys.argv.index("--sport")
        if idx + 1 < len(sys.argv):
            sport = sys.argv[idx + 1]

    print("=" * 60)
    print(f"LIVE PIPELINE - {VALIDATED_LEAGUES.get(sport, sport)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Fetch odds
    events = fetch_odds(sport)
    print(f"   {len(events)} matchs recuperes")

    # 2. Analyze each match
    picks = []
    for event in events:
        result = analyze_match(event)
        if result and result["score"] >= 4:
            picks.append(result)

    print(f"   {len(picks)} signaux Draw (Score >= 4)")

    # 3. Generate bet slips
    slips = generate_betslips(picks)

    # 4. Display
    display_results(picks, slips)

    # 5. Save recommendations
    output = {
        "timestamp": datetime.now().isoformat(),
        "sport": sport,
        "n_matches": len(events),
        "n_picks": len(picks),
        "picks": picks,
        "slips": [{"type": s["type"], "combined_odds": s["combined_odds"],
                    "kelly": s["kelly"],
                    "matches": [f"{l['home']} vs {l['away']}" for l in s["legs"]]}
                  for s in slips],
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/live_recommendations.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n   Recommandations sauvegardees : data/live_recommendations.json")


if __name__ == "__main__":
    main()
