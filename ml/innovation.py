"""
innovation.py - Recherche de breakthrough : meta-learning sur les signaux de marche

CE QUE PERSONNE NE FAIT :
Au lieu d'utiliser le ML pour predire les resultats de matchs,
on l'utilise pour predire QUELS SIGNAUX DE MISPRICING sont reels.

C'est un changement de paradigme :
- Standard : features du match -> prediction H/D/A
- Innovation : features du MISPRICING -> prediction profitable/non-profitable

APPROCHES TESTEES :
1. Meta-classifier : XGBoost apprend a filtrer les signaux Kaunitz
2. Topologie du desaccord : la FORME du desaccord predit la profitabilite
3. Biais par bookmaker : patterns d'erreurs systematiques par bookmaker
4. Seuil dynamique : seuil optimal par ligue/type/range de cotes
5. Regime detector : identifier les periodes d'inefficience du marche

Usage :
    python ml/innovation.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
from scipy.stats import entropy, skew
import warnings
warnings.filterwarnings("ignore")


def load_data():
    """Charge les donnees brutes."""
    dfs = []
    for f in sorted(Path("data/raw").glob("*.csv")):
        try:
            d = pd.read_csv(f, encoding="latin-1")
            parts = f.stem.split("_")
            d["league_code"] = parts[0]
            d["season"] = "_".join(parts[1:])
            dfs.append(d)
        except:
            pass
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"HomeTeam":"home_team","AwayTeam":"away_team",
                             "FTHG":"home_goals","FTAG":"away_goals","FTR":"result"})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date","result"]).sort_values("date").reset_index(drop=True)

    odds_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["B365","BW","IW","PS","WH","VC","Max","Avg"])]
    for c in odds_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Elo
    elo = {}
    K, HA = 25, 80
    elo_diffs, elo_close = [], []
    for _, r in df.iterrows():
        ht, at = r["home_team"], r["away_team"]
        diff = elo.get(ht, 1500) - elo.get(at, 1500)
        elo_diffs.append(diff)
        elo_close.append(1 / (1 + abs(diff)/100))
        hr = elo.get(ht,1500) + HA
        ar = elo.get(at,1500)
        e = 1/(1+10**((ar-hr)/400))
        s = 1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht] = elo.get(ht,1500) + K*(s-e)
        elo[at] = elo.get(at,1500) + K*((1-s)-(1-e))
    df["elo_diff"] = elo_diffs
    df["elo_closeness"] = elo_close

    return df


def extract_mispricing_signals(df):
    """
    Extraire TOUS les signaux de mispricing avec des meta-features.

    Pour chaque pari potentiel, on calcule des features sur le
    MISPRICING LUI-MEME, pas sur le match.
    """
    bk_h = [c for c in ["B365H","BWH","IWH","PSH","WHH","VCH"] if c in df.columns]
    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    bk_a = [c for c in ["B365A","BWA","IWA","PSA","WHA","VCA"] if c in df.columns]

    signals = []

    for idx, match in df.iterrows():
        for outcome, bk_cols, result_check in [
            ("H", bk_h, match["result"]=="H"),
            ("D", bk_d, match["result"]=="D"),
            ("A", bk_a, match["result"]=="A"),
        ]:
            odds_dict = {}
            for col in bk_cols:
                val = match.get(col)
                if pd.notna(val) and val > 1.0:
                    odds_dict[col] = val

            if len(odds_dict) < 3:
                continue

            odds_values = list(odds_dict.values())
            implied_probs = [1/o for o in odds_values]

            consensus_prob = np.mean(implied_probs)
            fair_odds = 1 / consensus_prob

            # Chercher la meilleure cote (potentiellement mispriced)
            best_bk = max(odds_dict, key=odds_dict.get)
            best_odds = odds_dict[best_bk]

            # Calculer l'edge par rapport au consensus
            edge = consensus_prob - 1/best_odds

            if edge < 0.01:  # Seuil minimum
                continue

            # ================================================
            # META-FEATURES : proprietes du MISPRICING
            # ================================================

            # 1. Taille du mispricing (edge)
            f_edge = edge

            # 2. Ratio best/fair
            f_ratio = best_odds / fair_odds

            # 3. Ecart-type des cotes (dispersion entre bookmakers)
            f_odds_std = np.std(odds_values)

            # 4. Coefficient de variation
            f_odds_cv = np.std(odds_values) / np.mean(odds_values)

            # 5. Skewness des cotes (asymetrie de la distribution)
            f_skew = skew(odds_values) if len(odds_values) >= 3 else 0

            # 6. Nombre de bookmakers "d'accord" avec le consensus
            # vs nombre d'outliers
            median_odds = np.median(odds_values)
            threshold_agree = 0.1 * median_odds
            n_agree = sum(1 for o in odds_values if abs(o - median_odds) < threshold_agree)
            f_agreement_ratio = n_agree / len(odds_values)

            # 7. Le bookmaker outlier est-il Pinnacle ?
            # (si Pinnacle est l'outlier, c'est probablement lui qui a raison)
            f_pinnacle_is_outlier = 1 if best_bk.startswith("PS") else 0

            # 8. Distance entre best et 2nd best
            sorted_odds = sorted(odds_values, reverse=True)
            f_gap_to_second = sorted_odds[0] - sorted_odds[1] if len(sorted_odds) > 1 else 0

            # 9. Pinnacle implied prob (si disponible)
            pin_col = {"H":"PSH","D":"PSD","A":"PSA"}[outcome]
            pin_odds = match.get(pin_col)
            f_pin_implied = 1/pin_odds if pd.notna(pin_odds) and pin_odds > 1 else consensus_prob

            # 10. Divergence Pinnacle vs consensus
            f_pin_vs_consensus = abs(f_pin_implied - consensus_prob)

            # 11. Le best est-il un soft bookmaker connu ?
            soft_books = ["B365", "BW", "WH", "VC"]
            f_is_soft = 1 if any(best_bk.startswith(s) for s in soft_books) else 0

            # 12. Contexte du match
            f_elo_closeness = match.get("elo_closeness", 0.5)
            f_elo_diff = abs(match.get("elo_diff", 0))

            # 13. Type de pari encode
            f_is_draw = 1 if outcome == "D" else 0
            f_is_away = 1 if outcome == "A" else 0

            # 14. Ligue (certaines ligues sont plus inefficientes)
            league = match.get("league_code", "")
            f_is_bundesliga = 1 if league == "D1" else 0
            f_is_ligue1 = 1 if league == "F1" else 0
            f_is_seriea = 1 if league == "I1" else 0
            f_is_epl = 1 if league == "E0" else 0
            f_is_laliga = 1 if league == "SP1" else 0

            # 15. Range de cotes (haute cote = plus de variance)
            f_odds_range = 0  # low
            if best_odds > 5:
                f_odds_range = 2  # high
            elif best_odds > 3:
                f_odds_range = 1  # medium

            # 16. Entropie de la distribution des cotes
            probs_norm = np.array(implied_probs) / sum(implied_probs)
            f_entropy = entropy(probs_norm)

            # TARGET : est-ce que ce pari est gagne ?
            won = result_check

            signals.append({
                # Meta-features
                "edge": f_edge,
                "ratio": f_ratio,
                "odds_std": f_odds_std,
                "odds_cv": f_odds_cv,
                "skewness": f_skew,
                "agreement": f_agreement_ratio,
                "pin_is_outlier": f_pinnacle_is_outlier,
                "gap_to_second": f_gap_to_second,
                "pin_implied": f_pin_implied,
                "pin_vs_consensus": f_pin_vs_consensus,
                "is_soft": f_is_soft,
                "elo_closeness": f_elo_closeness,
                "elo_diff_abs": f_elo_diff,
                "is_draw": f_is_draw,
                "is_away": f_is_away,
                "is_bundesliga": f_is_bundesliga,
                "is_ligue1": f_is_ligue1,
                "is_seriea": f_is_seriea,
                "is_epl": f_is_epl,
                "is_laliga": f_is_laliga,
                "odds_range": f_odds_range,
                "odds_entropy": f_entropy,
                # Info pour le backtest
                "best_odds": best_odds,
                "best_bk": best_bk,
                "outcome": outcome,
                "won": won,
                "match": f"{match['home_team']} vs {match['away_team']}",
                "league": league,
                "season": match.get("season",""),
                "consensus_prob": consensus_prob,
            })

    return pd.DataFrame(signals)


META_FEATURES = [
    "edge", "ratio", "odds_std", "odds_cv", "skewness",
    "agreement", "pin_is_outlier", "gap_to_second",
    "pin_implied", "pin_vs_consensus", "is_soft",
    "elo_closeness", "elo_diff_abs",
    "is_draw", "is_away",
    "is_bundesliga", "is_ligue1", "is_seriea", "is_epl", "is_laliga",
    "odds_range", "odds_entropy",
]


def backtest(signals_df, stake=10.0):
    """Backtest d'un ensemble de signaux."""
    if signals_df.empty:
        return {"n": 0, "roi": 0, "profit": 0}
    n = len(signals_df)
    profit = 0
    for _, s in signals_df.iterrows():
        if s["won"]:
            profit += stake * (s["best_odds"] - 1)
        else:
            profit -= stake
    return {
        "n": n,
        "roi": profit / (n * stake) * 100,
        "profit": profit,
        "win_rate": signals_df["won"].mean(),
    }


def print_backtest(label, signals_df, by_season=True):
    """Affiche les resultats."""
    r = backtest(signals_df)
    print(f"\n   {label}")
    print(f"   Paris: {r['n']}, Win: {r.get('win_rate',0):.1%}, "
          f"ROI: {r['roi']:+.1f}%, Profit: {r['profit']:+.1f}")

    if by_season and not signals_df.empty:
        seasons = sorted(signals_df["season"].unique())
        pos = 0
        for s in seasons:
            sub = signals_df[signals_df["season"] == s]
            sr = backtest(sub)
            if sr["n"] >= 5:
                m = " $$$" if sr["roi"] > 0 else ""
                if sr["roi"] > 0:
                    pos += 1
                print(f"     {s}: {sr['n']} paris, ROI {sr['roi']:+.1f}%{m}")
        print(f"     Saisons + : {pos}/{len(seasons)}")

    return r


# ============================================================
# INNOVATION 1 : META-CLASSIFIER
# ============================================================

def innovation_meta_classifier(signals_df):
    """
    XGBoost comme META-CLASSIFIER sur les signaux de mispricing.

    Target : est-ce que ce signal de mispricing est PROFITABLE ?
    Pas "qui gagne le match", mais "ce mispricing est-il reel".
    
    On calcule la profitabilite comme : won * (odds-1) - (1-won) * 1 > 0
    """
    print("\n" + "=" * 60)
    print("INNOVATION 1 : META-CLASSIFIER SUR SIGNAUX KAUNITZ")
    print("=" * 60)

    df = signals_df.copy()
    # Target binaire : le pari est-il profitable ?
    df["profitable"] = (df["won"].astype(int) * (df["best_odds"] - 1) - 
                         (~df["won"]).astype(int)).apply(lambda x: 1 if x > 0 else 0)

    seasons = sorted(df["season"].unique())

    print(f"   Total signaux : {len(df)}")
    print(f"   % profitable : {df['profitable'].mean():.1%}")

    all_selected = []

    print(f"\n   Walk-forward meta-learning :")
    print(f"   {'Season':<15} {'Signaux':>8} {'Select':>7} {'ROI raw':>9} {'ROI meta':>9}")
    print(f"   {'-'*50}")

    for i in range(1, len(seasons)):
        train = df[df["season"].isin(seasons[:i])]
        test = df[df["season"] == seasons[i]]

        if len(train) < 100 or len(test) < 20:
            continue

        X_tr = train[META_FEATURES].fillna(0)
        y_tr = train["profitable"]
        X_te = test[META_FEATURES].fillna(0)

        # Entrainer le meta-classifier
        meta = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42, verbosity=0,
        )
        meta.fit(X_tr, y_tr)

        # Predire la probabilite de profitabilite
        meta_probs = meta.predict_proba(X_te)[:, 1]

        # Ne garder que les signaux avec haute proba de profitabilite
        test = test.copy()
        test["meta_prob"] = meta_probs

        # ROI sans filtre (tous les signaux Kaunitz)
        r_raw = backtest(test)

        # ROI avec filtre meta (top 30% des signaux)
        threshold = np.percentile(meta_probs, 70)
        selected = test[test["meta_prob"] >= threshold]
        r_meta = backtest(selected)

        m_raw = " " if r_raw["roi"] <= 0 else "$"
        m_meta = " " if r_meta["roi"] <= 0 else "$$$"

        print(f"   {seasons[i]:<15} {len(test):>8} {len(selected):>7} "
              f"{r_raw['roi']:>+8.1f}% {r_meta['roi']:>+8.1f}% {m_meta}")

        all_selected.append(selected)

    if all_selected:
        combined = pd.concat(all_selected)
        print(f"   {'-'*50}")
        r = backtest(combined)
        print(f"   {'TOTAL':<15} {'':>8} {len(combined):>7} {'':>9} {r['roi']:>+8.1f}%")
        print_backtest("Meta-classifier selectionnes", combined)

        # Feature importance du meta-classifier
        print(f"\n   Meta-features les plus importantes :")
        # Re-entrainer sur tout pour voir l'importance
        X_all = signals_df[META_FEATURES].fillna(0)
        y_all = signals_df["won"].astype(int)
        meta_full = XGBClassifier(n_estimators=100, max_depth=3, verbosity=0)
        meta_full.fit(X_all, y_all)
        for name, imp in sorted(zip(META_FEATURES, meta_full.feature_importances_),
                                 key=lambda x: x[1], reverse=True)[:10]:
            bar = "#" * int(imp * 40)
            print(f"   {name:<22} {imp:.3f} {bar}")


# ============================================================
# INNOVATION 2 : TOPOLOGIE DU DESACCORD
# ============================================================

def innovation_topology(signals_df):
    """
    Analyse la STRUCTURE du desaccord, pas juste sa taille.

    Hypothese : quand le desaccord est "1 vs 5" (un seul outlier,
    les 5 autres d'accord), l'outlier a probablement tort = signal fiable.
    Quand c'est "3 vs 3", le marche est incertain = signal peu fiable.
    """
    print("\n" + "=" * 60)
    print("INNOVATION 2 : TOPOLOGIE DU DESACCORD")
    print("=" * 60)

    df = signals_df.copy()

    # Fort accord (agreement > 0.7) = 1 outlier vs le reste
    strong = df[df["agreement"] >= 0.7]
    weak = df[(df["agreement"] >= 0.3) & (df["agreement"] < 0.7)]

    print_backtest("Fort accord (1 outlier vs 5)", strong)
    print_backtest("Faible accord (marche divise)", weak)

    # Combiner : fort accord + draw + match equilibre
    best_topo = df[
        (df["agreement"] >= 0.7) &
        (df["is_draw"] == 1) &
        (df["elo_closeness"] >= 0.5)
    ]
    print_backtest("Fort accord + Draw + equilibre", best_topo)


# ============================================================
# INNOVATION 3 : BIAIS PAR BOOKMAKER
# ============================================================

def innovation_bookmaker_bias(signals_df):
    """
    Chaque bookmaker fait-il des erreurs systematiques ?
    
    Si Bet365 est systematiquement trop genereux sur les Draw
    en Serie A, c'est un biais exploitable.
    """
    print("\n" + "=" * 60)
    print("INNOVATION 3 : BIAIS PAR BOOKMAKER")
    print("=" * 60)

    df = signals_df.copy()
    stake = 10.0

    print(f"\n   ROI par bookmaker outlier :")
    print(f"   {'Bookmaker':<12} {'Paris':>6} {'Win%':>7} {'ROI':>8}")
    print(f"   {'-'*35}")

    for bk in sorted(df["best_bk"].unique()):
        sub = df[df["best_bk"] == bk]
        if len(sub) >= 20:
            r = backtest(sub)
            m = " $$$" if r["roi"] > 0 else ""
            print(f"   {bk:<12} {r['n']:>6} {r['win_rate']:>6.1%} {r['roi']:>+7.1f}%{m}")

    # Par bookmaker x ligue
    print(f"\n   Meilleurs combos bookmaker x ligue (> 20 paris) :")
    print(f"   {'Combo':<20} {'Paris':>6} {'ROI':>8}")
    print(f"   {'-'*36}")

    combos = []
    for bk in df["best_bk"].unique():
        for lg in df["league"].unique():
            sub = df[(df["best_bk"] == bk) & (df["league"] == lg)]
            if len(sub) >= 20:
                r = backtest(sub)
                combos.append({"combo": f"{bk}+{lg}", "n": r["n"], "roi": r["roi"]})

    combos.sort(key=lambda x: x["roi"], reverse=True)
    for c in combos[:10]:
        m = " $$$" if c["roi"] > 0 else ""
        print(f"   {c['combo']:<20} {c['n']:>6} {c['roi']:>+7.1f}%{m}")


# ============================================================
# INNOVATION 4 : SEUIL DYNAMIQUE OPTIMAL
# ============================================================

def innovation_dynamic_threshold(signals_df):
    """
    Au lieu d'un seuil fixe, trouve le seuil optimal
    pour chaque combinaison ligue/type.
    """
    print("\n" + "=" * 60)
    print("INNOVATION 4 : SEUIL DYNAMIQUE")
    print("=" * 60)

    df = signals_df.copy()
    stake = 10.0

    print(f"\n   ROI par range d'edge :")
    print(f"   {'Edge range':<15} {'Paris':>6} {'Win%':>7} {'ROI':>8} {'Avg odds':>9}")
    print(f"   {'-'*47}")

    for lo, hi, label in [
        (0.01, 0.03, "1-3%"),
        (0.03, 0.05, "3-5%"),
        (0.05, 0.08, "5-8%"),
        (0.08, 0.12, "8-12%"),
        (0.12, 0.20, "12-20%"),
        (0.20, 1.00, "20%+"),
    ]:
        sub = df[(df["edge"] >= lo) & (df["edge"] < hi)]
        if len(sub) >= 20:
            r = backtest(sub)
            m = " $$$" if r["roi"] > 0 else ""
            print(f"   {label:<15} {r['n']:>6} {r['win_rate']:>6.1%} "
                  f"{r['roi']:>+7.1f}% {sub['best_odds'].mean():>8.2f}{m}")

    # Optimal par ligue x type
    print(f"\n   Seuil optimal par niche :")
    print(f"   {'Niche':<25} {'Best edge':>10} {'Paris':>6} {'ROI':>8}")
    print(f"   {'-'*51}")

    for lg in sorted(df["league"].unique()):
        for otype in ["H", "D", "A"]:
            sub = df[(df["league"] == lg) & (df["outcome"] == otype)]
            if len(sub) < 30:
                continue

            best_roi = -999
            best_edge = 0
            for edge_cut in np.arange(0.02, 0.25, 0.02):
                filtered = sub[sub["edge"] >= edge_cut]
                if len(filtered) >= 10:
                    r = backtest(filtered)
                    if r["roi"] > best_roi:
                        best_roi = r["roi"]
                        best_edge = edge_cut
                        best_n = r["n"]

            if best_roi > -999:
                m = " $$$" if best_roi > 0 else ""
                name = {"H":"Home","D":"Draw","A":"Away"}[otype]
                print(f"   {lg} {name:<19} {best_edge:>9.0%} {best_n:>6} {best_roi:>+7.1f}%{m}")


# ============================================================
# INNOVATION 5 : COMBINAISON OPTIMALE
# ============================================================

def innovation_combined(signals_df):
    """
    Combine les meilleurs filtres decouverts.
    """
    print("\n" + "=" * 60)
    print("INNOVATION 5 : COMBINAISON DES DECOUVERTES")
    print("=" * 60)

    df = signals_df.copy()

    # Tester differentes combinaisons de filtres
    filters = [
        ("Draw + equilibre + fort accord",
         (df["is_draw"]==1) & (df["elo_closeness"]>=0.5) & (df["agreement"]>=0.7)),
        ("Draw + equilibre + edge>5%",
         (df["is_draw"]==1) & (df["elo_closeness"]>=0.5) & (df["edge"]>=0.05)),
        ("Draw + fort accord + soft book",
         (df["is_draw"]==1) & (df["agreement"]>=0.7) & (df["is_soft"]==1)),
        ("Draw + Pin pas outlier + equilibre",
         (df["is_draw"]==1) & (df["pin_is_outlier"]==0) & (df["elo_closeness"]>=0.5)),
        ("Tout sauf LaLiga + Draw + equilibre",
         (df["is_draw"]==1) & (df["is_laliga"]==0) & (df["elo_closeness"]>=0.5)),
        ("Bundesliga + Draw",
         (df["is_draw"]==1) & (df["is_bundesliga"]==1)),
        ("SerieA + Draw + edge>5%",
         (df["is_draw"]==1) & (df["is_seriea"]==1) & (df["edge"]>=0.05)),
        ("Fort accord + pas Pin outlier",
         (df["agreement"]>=0.7) & (df["pin_is_outlier"]==0)),
        ("Edge 5-15% + fort accord + Draw",
         (df["edge"]>=0.05) & (df["edge"]<0.15) & (df["agreement"]>=0.7) & (df["is_draw"]==1)),
        ("Gap to 2nd > 0.5 + Draw",
         (df["gap_to_second"]>=0.5) & (df["is_draw"]==1)),
    ]

    print(f"\n   {'Filtre':<45} {'Paris':>6} {'ROI':>8} {'S+':>4}")
    print(f"   {'-'*65}")

    for label, mask in filters:
        sub = df[mask]
        if len(sub) >= 20:
            r = backtest(sub)
            # Compter saisons positives
            pos = 0
            for s in sub["season"].unique():
                sr = backtest(sub[sub["season"]==s])
                if sr["n"] >= 5 and sr["roi"] > 0:
                    pos += 1
            total_s = len(sub["season"].unique())
            m = " $$$" if r["roi"] > 0 else ""
            print(f"   {label:<45} {r['n']:>6} {r['roi']:>+7.1f}% {pos}/{total_s}{m}")


def main():
    print("=" * 60)
    print("INNOVATION : META-LEARNING SUR SIGNAUX DE MARCHE")
    print("=" * 60)

    df = load_data()
    print(f"\n   Extraction des signaux de mispricing...")
    signals = extract_mispricing_signals(df)
    print(f"   {len(signals)} signaux extraits")
    print(f"   Win rate brut : {signals['won'].mean():.1%}")

    innovation_meta_classifier(signals)
    innovation_topology(signals)
    innovation_bookmaker_bias(signals)
    innovation_dynamic_threshold(signals)
    innovation_combined(signals)

    print("\n" + "=" * 60)
    print("INNOVATION TERMINEE")
    print("=" * 60)


if __name__ == "__main__":
    main()
