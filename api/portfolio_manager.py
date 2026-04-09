"""
portfolio_manager.py - Gestion du portefeuille (paper trading)

Persiste l'etat dans un fichier JSON.
"""

import json
from pathlib import Path
from datetime import datetime


class PortfolioManager:
    def __init__(self, state_file="data/portfolio_state.json", initial_bankroll=1000):
        self.state_file = Path(state_file)
        self.initial_bankroll = initial_bankroll

        if self.state_file.exists():
            with open(self.state_file) as f:
                self.state = json.load(f)
        else:
            self.reset()

    def _save(self):
        self.state_file.parent.mkdir(exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def reset(self):
        self.state = {
            "bankroll": self.initial_bankroll,
            "initial_bankroll": self.initial_bankroll,
            "pending": [],
            "history": [],
            "total_bets": 0,
            "total_wins": 0,
            "total_profit": 0.0,
            "created_at": datetime.now().isoformat(),
        }
        self._save()

    def place_bet(self, match, bet_type, odds, stake, prob):
        if stake > self.state["bankroll"]:
            return {"error": "Bankroll insuffisant"}

        bet = {
            "id": self.state["total_bets"],
            "match": match,
            "bet_type": bet_type,
            "odds": odds,
            "stake": stake,
            "prob": prob,
            "edge": round(prob - 1/odds, 4),
            "placed_at": datetime.now().isoformat(),
            "status": "pending",
        }

        self.state["bankroll"] -= stake
        self.state["pending"].append(bet)
        self.state["total_bets"] += 1
        self._save()

        return {"message": "Pari place", "bet": bet, "bankroll": round(self.state["bankroll"], 2)}

    def resolve_bet(self, bet_id, won, closing_odds=None):
        # Trouver le pari
        for i, bet in enumerate(self.state["pending"]):
            if bet["id"] == bet_id:
                bet = self.state["pending"].pop(i)
                bet["status"] = "won" if won else "lost"
                bet["resolved_at"] = datetime.now().isoformat()

                if won:
                    profit = bet["stake"] * (bet["odds"] - 1)
                    self.state["bankroll"] += bet["stake"] + profit
                    self.state["total_wins"] += 1
                else:
                    profit = -bet["stake"]

                bet["profit"] = round(profit, 2)
                self.state["total_profit"] += profit

                if closing_odds:
                    bet["closing_odds"] = closing_odds
                    bet["clv"] = round(1/bet["odds"] - 1/closing_odds, 4)

                self.state["history"].append(bet)
                self._save()
                return bet

        return None

    def get_status(self):
        s = self.state
        n = s["total_bets"]
        return {
            "bankroll": round(s["bankroll"], 2),
            "initial_bankroll": s["initial_bankroll"],
            "total_bets": n,
            "pending": len(s["pending"]),
            "win_rate": round(s["total_wins"] / n, 3) if n > 0 else 0,
            "total_profit": round(s["total_profit"], 2),
            "roi": round(s["total_profit"] / s["initial_bankroll"] * 100, 1) if n > 0 else 0,
        }

    def get_history(self):
        return {"bets": self.state["history"], "count": len(self.state["history"])}

    def get_clv_summary(self):
        clv_bets = [b for b in self.state["history"] if "clv" in b]
        if not clv_bets:
            return {"message": "Pas encore de CLV", "count": 0}

        clvs = [b["clv"] for b in clv_bets]
        return {
            "count": len(clvs),
            "avg_clv": round(sum(clvs) / len(clvs), 4),
            "positive_pct": round(len([c for c in clvs if c > 0]) / len(clvs), 3),
            "best": round(max(clvs), 4),
            "worst": round(min(clvs), 4),
        }
