# 1TBET App

AI-powered sports betting portfolio manager — from data collection to deployed web app.

## Phase 1: Real Data + ML (current)

Trains an XGBoost model on **real historical match data** from football-data.co.uk (5 seasons, 5 leagues, 8,000+ matches) with Platt Scaling calibration and +EV backtesting.

### Quick Start

```bash
pip install -r requirements.txt
python download_data.py    # Download 5 seasons from 5 leagues
python train.py            # Train XGBoost + calibrate + backtest
```

### Data Source

[football-data.co.uk](https://www.football-data.co.uk/) provides free CSV files with match results and bookmaker odds since 1993. We use:
- Premier League, La Liga, Ligue 1, Bundesliga, Serie A
- Seasons 2020-2021 through 2024-2025
- Bet365 and Pinnacle odds for implied probability features

### Features

| Feature | Source | Description |
|---------|--------|-------------|
| Elo rating + diff | Calculated | Dynamic team strength |
| Home/away form | Calculated | Goals scored last 5 matches |
| Defense form | Calculated | Goals conceded last 5 matches |
| Bet365 implied probs | football-data | Normalized market consensus |
| Pinnacle implied probs | football-data | Sharp market reference |
| Odds difference | football-data | Market confidence signal |

### Roadmap

- [x] Phase 1: Real data + ML model
- [ ] Phase 2: FastAPI backend
- [ ] Phase 3: Frontend dashboard
- [ ] Phase 4: Deployment

## License

MIT
