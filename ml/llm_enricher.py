"""
llm_enricher.py - Couche LLM (Gemini gratuit) pour filtrer les paris

Le LLM ne predit PAS le resultat. Il cherche des RED FLAGS contextuels.
Signal statistique fort + absence de red flags = pari ultime.

Usage :
    set GEMINI_API_KEY=votre-cle
    python ml/llm_enricher.py
"""

import os
import json
import requests
from datetime import datetime


GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"

SYSTEM_PROMPT = """Tu es un analyste football expert. On te donne un match avec un signal statistique de mispricing sur le Draw.

Evalue si le CONTEXTE est favorable ou defavorable au Draw.

Reponds UNIQUEMENT en JSON valide (pas de markdown, pas de backticks) :
{"confidence": 0.0 a 1.0, "red_flags": ["liste"], "green_flags": ["liste"], "verdict": "BET" ou "SKIP", "reasoning": "court"}

FAVORISE LE DRAW : equipes de niveau similaire, milieu de classement, equipes defensives, historique de draws, matchs sans enjeu.
RED FLAGS (SKIP) : joueur star absent, derby, equipe en grande forme, changement coach, gros enjeu (titre/relegation).
Sois conservateur : doute = SKIP. confidence < 0.5 = SKIP."""


def enrich_bet(home, away, league, odds, edge, context=""):
    if not GEMINI_KEY:
        return _demo(home, away, league, odds, edge)

    prompt = f"""{SYSTEM_PROMPT}

Match : {home} vs {away}
Ligue : {league}
Cote Draw : {odds}
Edge statistique : {edge:.1%}
{context}

Reponds en JSON uniquement."""

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30,
        )
        if resp.status_code == 200:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            text = text.strip().strip("`").strip()
            if text.startswith("json"): text = text[4:].strip()
            return json.loads(text)
        else:
            print(f"   Gemini error {resp.status_code}: {resp.text[:200]}")
            return _demo(home, away, league, odds, edge)
    except Exception as e:
        print(f"   Error: {e}")
        return _demo(home, away, league, odds, edge)


def enrich_batch(bets):
    import time
    results = []
    for b in bets:
        r = enrich_bet(b["home_team"], b["away_team"],
                       b.get("league",""), b["odds"], b["edge"],
                       b.get("context",""))
        results.append({**b, **r})
        time.sleep(2)  # 2 secondes entre chaque appel
    return results


def filter_bets(enriched, min_conf=0.5):
    return [b for b in enriched if b.get("confidence", 0) >= min_conf]


def _demo(home, away, league, odds, edge):
    red, green = [], []
    conf = 0.6
    bigs = ["Barcelona","Real Madrid","Bayern","Man City","PSG","Juventus",
            "Liverpool","Arsenal","Inter","Chelsea","Man United","Dortmund"]
    hb = any(t in home for t in bigs)
    ab = any(t in away for t in bigs)
    if hb and ab:
        green.append("Choc entre grandes equipes")
        conf += 0.1
    elif hb or ab:
        red.append("Desequilibre grande vs petite equipe")
        conf -= 0.15
    if 3.0 <= odds <= 4.0:
        green.append("Cotes Draw optimales")
        conf += 0.05
    if edge > 0.05:
        green.append(f"Edge fort ({edge:.1%})")
        conf += 0.05
    conf = max(0.0, min(1.0, conf))
    return {"confidence": round(conf, 2), "red_flags": red, "green_flags": green,
            "verdict": "BET" if conf >= 0.5 else "SKIP",
            "reasoning": f"Confiance {conf:.0%} (mode demo)"}


if __name__ == "__main__":
    print("=" * 60)
    print("TEST LLM ENRICHER (Gemini)")
    print("=" * 60)

    tests = [
        {"home_team":"Everton","away_team":"Newcastle","league":"Premier League","odds":3.40,"edge":0.06},
        {"home_team":"Barcelona","away_team":"Atletico Madrid","league":"La Liga","odds":4.20,"edge":0.04},
        {"home_team":"Frosinone","away_team":"Sassuolo","league":"Serie B","odds":3.10,"edge":0.07},
        {"home_team":"Leganes","away_team":"Eibar","league":"La Liga 2","odds":3.00,"edge":0.05},
        {"home_team":"Man City","away_team":"Burnley","league":"Premier League","odds":8.50,"edge":0.03},
    ]

    enriched = enrich_batch(tests)
    filtered = filter_bets(enriched)

    print(f"\n   {len(tests)} paris analyses, {len(filtered)} gardes")
    print(f"   Mode: {'GEMINI API' if GEMINI_KEY else 'DEMO'}\n")

    for b in enriched:
        s = "BET " if b["verdict"] == "BET" else "SKIP"
        print(f"   [{s}] {b['home_team']} vs {b['away_team']} (conf={b['confidence']:.0%})")
        for f in b.get("green_flags", []):
            print(f"     + {f}")
        for f in b.get("red_flags", []):
            print(f"     - {f}")
        if b.get("reasoning"):
            print(f"     > {b['reasoning']}")
