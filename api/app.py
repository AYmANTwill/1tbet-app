"""
app.py - Backend FastAPI pour 1TBET

Endpoints :
  GET  /                  -> Status de l'API
  GET  /matches/today     -> Matchs du jour avec predictions
  GET  /predictions       -> Predictions du modele sur les matchs a venir
  GET  /bets/recommended  -> Paris recommandes (Draw +EV)
  GET  /portfolio/status  -> Etat du portefeuille
  POST /portfolio/place   -> Placer un pari (paper trading)
  GET  /clv               -> Closing Line Value tracker
  GET  /odds/live         -> Cotes live depuis The Odds API

Usage :
    uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Import des modules locaux
from odds_client import fetch_live_odds, get_upcoming_matches
from predictor import predict_matches, find_ev_bets
from portfolio_manager import PortfolioManager

app = FastAPI(
    title="1TBET - AI Sports Betting API",
    description="Backend pour le portfolio manager de paris sportifs",
    version="1.0.0",
)

# Autoriser le frontend a appeler l'API (CORS)
# Sans ca, un navigateur bloquerait les requetes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, mettre l'URL du frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le portfolio au demarrage
portfolio = PortfolioManager()


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Status de l'API."""
    return {
        "status": "running",
        "name": "1TBET API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "model": "XGBoost + Platt Scaling (Draw specialist)",
    }


@app.get("/odds/live")
def get_live_odds(sport: str = "soccer_epl"):
    """
    Recupere les cotes live depuis The Odds API.

    Params :
        sport : cle du sport (soccer_epl, soccer_spain_la_liga, etc.)
    """
    try:
        odds = fetch_live_odds(sport)
        return {"sport": sport, "matches": odds, "count": len(odds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
def get_predictions(sport: str = "soccer_epl"):
    """
    Genere les predictions du modele sur les matchs a venir.
    Combine les cotes live avec le modele XGBoost.
    """
    try:
        matches = get_upcoming_matches(sport)
        if not matches:
            return {"predictions": [], "message": "Aucun match a venir"}

        predictions = predict_matches(matches)
        return {"sport": sport, "predictions": predictions, "count": len(predictions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bets/recommended")
def get_recommended_bets(
    sport: str = "soccer_epl",
    min_edge: float = 0.12,
    draw_only: bool = True,
):
    """
    Paris recommandes par le modele.

    Params :
        sport : ligue ciblee
        min_edge : edge minimum (12% par defaut)
        draw_only : ne recommander que les Draw (meilleur ROI)
    """
    try:
        matches = get_upcoming_matches(sport)
        if not matches:
            return {"bets": [], "message": "Aucun match a venir"}

        predictions = predict_matches(matches)
        recommended = find_ev_bets(predictions, min_edge=min_edge, draw_only=draw_only)

        return {
            "sport": sport,
            "min_edge": min_edge,
            "draw_only": draw_only,
            "bets": recommended,
            "count": len(recommended),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/status")
def get_portfolio_status():
    """Retourne l'etat actuel du portefeuille."""
    return portfolio.get_status()


@app.post("/portfolio/place")
def place_bet(match: str, bet_type: str, odds: float, stake: float, prob: float):
    """
    Place un pari en paper trading.

    Params :
        match : nom du match ("Arsenal vs Chelsea")
        bet_type : "H", "D", ou "A"
        odds : cote decimale
        stake : mise en unites
        prob : probabilite du modele
    """
    if bet_type not in ["H", "D", "A"]:
        raise HTTPException(status_code=400, detail="bet_type doit etre H, D, ou A")
    if stake <= 0:
        raise HTTPException(status_code=400, detail="stake doit etre positif")

    result = portfolio.place_bet(match, bet_type, odds, stake, prob)
    return result


@app.post("/portfolio/resolve/{bet_id}")
def resolve_bet(bet_id: int, won: bool, closing_odds: float = None):
    """Resout un pari (gagne ou perdu)."""
    result = portfolio.resolve_bet(bet_id, won, closing_odds)
    if result is None:
        raise HTTPException(status_code=404, detail="Pari non trouve")
    return result


@app.get("/portfolio/history")
def get_history():
    """Historique de tous les paris."""
    return portfolio.get_history()


@app.get("/clv")
def get_clv():
    """Resume du Closing Line Value."""
    return portfolio.get_clv_summary()


@app.post("/portfolio/reset")
def reset_portfolio():
    """Remet le portefeuille a zero."""
    portfolio.reset()
    return {"message": "Portfolio reset", "bankroll": 1000}
