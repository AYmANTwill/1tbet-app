"""
download_data.py - Telecharge les donnees historiques de football-data.co.uk

FOOTBALL-DATA.CO.UK :
C'est LA source gratuite de reference pour les donnees de football.
Ils fournissent des fichiers CSV avec :
- Les resultats de chaque match (score, date, equipes)
- Les cotes de 10+ bookmakers (Bet365, Pinnacle, etc.)
- Les stats de match (tirs, corners, cartons)
- Couverture de 25+ ligues depuis 1993

FORMAT DES COLONNES PRINCIPALES :
    Div      : code de la ligue (E0 = Premier League, SP1 = La Liga, etc.)
    Date     : date du match
    HomeTeam : equipe a domicile
    AwayTeam : equipe a l'exterieur
    FTHG     : buts domicile (Full Time Home Goals)
    FTAG     : buts exterieur (Full Time Away Goals)
    FTR      : resultat (H = Home, D = Draw, A = Away)
    B365H    : cote Bet365 domicile
    B365D    : cote Bet365 nul
    B365A    : cote Bet365 exterieur
    PSH/PSD/PSA : cotes Pinnacle (le bookmaker le plus sharp)

Usage :
    python download_data.py
"""

import os
import requests
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION : quelles ligues et saisons telecharger
# ============================================================

# Codes des ligues sur football-data.co.uk
LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "F1": "Ligue 1",
    "D1": "Bundesliga",
    "I1": "Serie A",
}

# Saisons a telecharger (format YYYY pour debut de saison)
# 2021 = saison 2021-2022, etc.
SEASONS = [2020, 2021, 2022, 2023, 2024]

# URL de base
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Dossier de sortie
DATA_DIR = Path("data/raw")


def build_url(season_start, league_code):
    """
    Construit l'URL de telechargement d'un CSV.

    Format : https://www.football-data.co.uk/mmz4281/XXYY/CODE.csv
    XX = deux derniers chiffres de l'annee de debut
    YY = deux derniers chiffres de l'annee de fin

    Exemple : saison 2023-2024, Premier League
    -> https://www.football-data.co.uk/mmz4281/2324/E0.csv
    """
    start = str(season_start)[2:]  # 2023 -> "23"
    end = str(season_start + 1)[2:]  # 2024 -> "24"
    return f"{BASE_URL}/{start}{end}/{league_code}.csv"


def download_csv(url, filepath):
    """
    Telecharge un fichier CSV depuis une URL.

    requests.get() envoie une requete HTTP GET (comme un navigateur).
    response.status_code == 200 signifie "succes".
    On ecrit le contenu brut dans un fichier.
    """
    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"   Erreur {response.status_code} pour {url}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"   Erreur de connexion : {e}")
        return False


def download_all():
    """
    Telecharge toutes les ligues et saisons configurees.
    """
    print("=" * 60)
    print("TELECHARGEMENT DES DONNEES HISTORIQUES")
    print("=" * 60)
    print(f"   Ligues : {', '.join(LEAGUES.values())}")
    print(f"   Saisons : {SEASONS[0]}-{SEASONS[0]+1} a {SEASONS[-1]}-{SEASONS[-1]+1}")

    # Creer le dossier
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    total_failed = 0

    for league_code, league_name in LEAGUES.items():
        print(f"\n   {league_name} ({league_code}) :")

        for season in SEASONS:
            url = build_url(season, league_code)
            filename = f"{league_code}_{season}_{season+1}.csv"
            filepath = DATA_DIR / filename

            # Ne pas re-telecharger si le fichier existe deja
            if filepath.exists():
                print(f"   [{season}-{season+1}] Deja telecharge")
                total_downloaded += 1
                continue

            print(f"   [{season}-{season+1}] Telechargement...", end=" ")
            success = download_csv(url, filepath)

            if success:
                # Verifier que le CSV est valide
                try:
                    df = pd.read_csv(filepath, encoding="latin-1")
                    print(f"OK ({len(df)} matchs)")
                    total_downloaded += 1
                except Exception:
                    print("CSV invalide, supprime")
                    filepath.unlink()
                    total_failed += 1
            else:
                total_failed += 1

    print(f"\n   Telecharges : {total_downloaded}")
    print(f"   Echecs : {total_failed}")
    print(f"   Fichiers dans : {DATA_DIR}/")


def merge_all():
    """
    Fusionne tous les CSV en un seul DataFrame propre.

    On garde uniquement les colonnes utiles et on standardise les noms.
    """
    print("\n" + "=" * 60)
    print("FUSION DES DONNEES")
    print("=" * 60)

    all_dfs = []

    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, encoding="latin-1")

            # Identifier la ligue et la saison depuis le nom du fichier
            parts = csv_file.stem.split("_")
            league_code = parts[0]
            season = parts[1]

            # Colonnes essentielles qu'on garde
            # Certaines saisons ont des noms de colonnes differents
            cols_needed = {
                "Div": "league",
                "Date": "date",
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
                "FTR": "result",
            }

            # Colonnes de cotes (on prend Bet365 et Pinnacle)
            odds_cols = {}
            if "B365H" in df.columns:
                odds_cols.update({
                    "B365H": "b365_home",
                    "B365D": "b365_draw",
                    "B365A": "b365_away",
                })
            if "PSH" in df.columns:
                odds_cols.update({
                    "PSH": "pinnacle_home",
                    "PSD": "pinnacle_draw",
                    "PSA": "pinnacle_away",
                })
            # Cotes Pinnacle parfois sous un autre nom
            elif "PH" in df.columns:
                odds_cols.update({
                    "PH": "pinnacle_home",
                    "PD": "pinnacle_draw",
                    "PA": "pinnacle_away",
                })

            # Verifier que les colonnes essentielles existent
            available = {k: v for k, v in cols_needed.items() if k in df.columns}
            if len(available) < 7:
                print(f"   {csv_file.name} : colonnes manquantes, ignore")
                continue

            # Renommer et selectionner
            rename_map = {**cols_needed, **odds_cols}
            available_rename = {k: v for k, v in rename_map.items() if k in df.columns}
            df_clean = df[list(available_rename.keys())].rename(columns=available_rename)

            # Ajouter la saison
            df_clean["season"] = f"{season}-{int(season)+1}"

            # Ajouter le nom lisible de la ligue
            df_clean["league_name"] = LEAGUES.get(league_code, league_code)

            all_dfs.append(df_clean)
            print(f"   {csv_file.name} : {len(df_clean)} matchs")

        except Exception as e:
            print(f"   {csv_file.name} : erreur ({e})")

    if not all_dfs:
        print("   Aucune donnee chargee !")
        return None

    # Fusionner tous les DataFrames
    merged = pd.concat(all_dfs, ignore_index=True)

    # Nettoyer les dates
    # football-data.co.uk utilise DD/MM/YYYY
    merged["date"] = pd.to_datetime(merged["date"], dayfirst=True, format="mixed")

    # Trier par date
    merged = merged.sort_values("date").reset_index(drop=True)

    # Supprimer les lignes avec des cotes manquantes
    before = len(merged)
    merged = merged.dropna(subset=["b365_home", "b365_away"]).reset_index(drop=True)
    after = len(merged)
    if before != after:
        print(f"\n   {before - after} lignes supprimees (cotes manquantes)")

    # Sauvegarder
    output_path = Path("data/matches_all.csv")
    merged.to_csv(output_path, index=False)

    print(f"\n   TOTAL : {len(merged)} matchs")
    print(f"   Ligues : {merged['league_name'].nunique()}")
    print(f"   Periode : {merged['date'].min().date()} a {merged['date'].max().date()}")
    print(f"   Sauvegarde : {output_path}")

    # Stats par ligue
    print(f"\n   {'Ligue':<20} {'Matchs':>8} {'H%':>6} {'D%':>6} {'A%':>6}")
    print(f"   {'-'*48}")
    for league in sorted(merged["league_name"].unique()):
        sub = merged[merged["league_name"] == league]
        h = (sub["result"] == "H").mean()
        d = (sub["result"] == "D").mean()
        a = (sub["result"] == "A").mean()
        print(f"   {league:<20} {len(sub):>8} {h:>5.1%} {d:>5.1%} {a:>5.1%}")

    return merged


if __name__ == "__main__":
    download_all()
    merge_all()
