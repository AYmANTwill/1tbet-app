"""
odds_client.py - Client pour The Odds API (cotes live)

Recupere les cotes en temps reel depuis The Odds API.
Necessite une cle API dans le fichier .env
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"

SPORTS_MAP = {
    "soccer_epl": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_france_ligue_one": "Ligue 1",
    "soccer_italy_serie_a": "Serie A",
    "soccer_germany_bundesliga": "Bundesliga",
}


def fetch_live_odds(sport="soccer_epl"):
    """Recupere les cotes live pour un sport."""
    if not API_KEY:
        return _demo_odds(sport)

    url = f"{BASE_URL}/sports/{sport}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "uk,eu",
        "oddsFormat": "decimal",
        "markets": "h2h",
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        return _demo_odds(sport)

    events = response.json()
    return _parse_events(events)


def _parse_events(events):
    """Parse la reponse API en format standard."""
    matches = []
    for event in events:
        bookmakers_odds = {}
        for bk in event.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    bookmakers_odds[bk["title"]] = {
                        "home": outcomes.get(event["home_team"]),
                        "draw": outcomes.get("Draw"),
                        "away": outcomes.get(event["away_team"]),
                    }

        # Trouver les meilleures cotes et la moyenne
        if bookmakers_odds:
            all_home = [b["home"] for b in bookmakers_odds.values() if b["home"]]
            all_draw = [b["draw"] for b in bookmakers_odds.values() if b["draw"]]
            all_away = [b["away"] for b in bookmakers_odds.values() if b["away"]]

            matches.append({
                "home_team": event["home_team"],
                "away_team": event["away_team"],
                "commence_time": event["commence_time"],
                "best_home_odds": max(all_home) if all_home else None,
                "best_draw_odds": max(all_draw) if all_draw else None,
                "best_away_odds": max(all_away) if all_away else None,
                "avg_home_odds": round(sum(all_home)/len(all_home), 2) if all_home else None,
                "avg_draw_odds": round(sum(all_draw)/len(all_draw), 2) if all_draw else None,
                "avg_away_odds": round(sum(all_away)/len(all_away), 2) if all_away else None,
                "n_bookmakers": len(bookmakers_odds),
            })

    return matches


def get_upcoming_matches(sport="soccer_epl"):
    """Retourne les matchs a venir avec cotes."""
    return fetch_live_odds(sport)


def _demo_odds(sport):
    """Donnees de demo quand pas de cle API."""
    return [
        {
            "home_team": "Arsenal", "away_team": "Chelsea",
            "commence_time": "2026-04-12T15:00:00Z",
            "best_home_odds": 1.85, "best_draw_odds": 3.60, "best_away_odds": 4.50,
            "avg_home_odds": 1.80, "avg_draw_odds": 3.50, "avg_away_odds": 4.30,
            "n_bookmakers": 25,
        },
        {
            "home_team": "Liverpool", "away_team": "Man United",
            "commence_time": "2026-04-12T17:30:00Z",
            "best_home_odds": 1.55, "best_draw_odds": 4.20, "best_away_odds": 5.80,
            "avg_home_odds": 1.50, "avg_draw_odds": 4.00, "avg_away_odds": 5.50,
            "n_bookmakers": 28,
        },
        {
            "home_team": "Everton", "away_team": "Newcastle",
            "commence_time": "2026-04-12T15:00:00Z",
            "best_home_odds": 2.90, "best_draw_odds": 3.30, "best_away_odds": 2.50,
            "avg_home_odds": 2.80, "avg_draw_odds": 3.20, "avg_away_odds": 2.45,
            "n_bookmakers": 22,
        },
        {
            "home_team": "Aston Villa", "away_team": "Brighton",
            "commence_time": "2026-04-12T15:00:00Z",
            "best_home_odds": 2.20, "best_draw_odds": 3.50, "best_away_odds": 3.40,
            "avg_home_odds": 2.15, "avg_draw_odds": 3.40, "avg_away_odds": 3.30,
            "n_bookmakers": 24,
        },
    ]
