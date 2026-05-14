"""
multi_sport_v2.py - Corrections + approfondissement multi-sport

CORRECTIONS :
- Tennis : utilise le vrai resultat (Winner gagne toujours dans tennis-data)
- Football : Kaunitz 6 signaux + momentum + points

APPROFONDISSEMENT :
- Tennis backtest + permutation test
- Football enrichi walk-forward
- NBA/UFC live + cross-sport slips

Usage :
    $env:ODDS_API_KEY="ta-cle"
    python ml/multi_sport_v2.py
"""

import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ATTENTION : Les donnees tennis-data.co.uk ont un DATA LEAKAGE
# B365W/PSW = cotes du GAGNANT (retrospectif). Inutilisable pour Kaunitz.
# Il faudrait des donnees avec cotes par Player1/Player2 (pas Winner/Loser).


def analyze_tennis():
    print("\n" + "=" * 60)
    print("TENNIS CORRIGE")
    print("=" * 60)

    raw = Path("data/raw_tennis")
    dfs = []
    for f in sorted(raw.glob("*.csv")):
        try:
            d = pd.read_csv(f, encoding="latin-1")
            d["source"] = f.stem
            dfs.append(d)
        except: pass
    if not dfs:
        print("   Pas de donnees tennis")
        return

    df = pd.concat(dfs, ignore_index=True)
    bk_w = [c for c in ["B365W","PSW","EXW","LBW","SJW"] if c in df.columns]
    bk_l = [c for c in ["B365L","PSL","EXL","LBL","SJL"] if c in df.columns]
    for c in bk_w+bk_l: df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"   {len(df)} matchs, BK_W={bk_w}, BK_L={bk_l}")

    stake = 10
    bets = []
    for _, m in df.iterrows():
        # Winner (toujours gagne)
        ow = {c:m[c] for c in bk_w if pd.notna(m.get(c)) and m.get(c)>1}
        if len(ow) >= 2:
            vals = list(ow.values())
            cons = np.mean([1/o for o in vals])
            bb = max(ow, key=ow.get); bo = ow[bb]
            edge = cons - 1/bo
            if edge >= 0.02:
                sc = sum([edge>=0.03, not bb.startswith("PS"),
                          sum(1 for o in vals if abs(o-np.median(vals))<0.1*np.median(vals))/len(vals)>=0.5,
                          bo > 1.5])
                bets.append({"type":"W","odds":bo,"edge":edge,"score":sc,
                             "won":True,"source":m.get("source","")})

        # Loser (toujours perd)
        ol = {c:m[c] for c in bk_l if pd.notna(m.get(c)) and m.get(c)>1}
        if len(ol) >= 2:
            vals = list(ol.values())
            cons = np.mean([1/o for o in vals])
            bb = max(ol, key=ol.get); bo = ol[bb]
            edge = cons - 1/bo
            if edge >= 0.02:
                sc = sum([edge>=0.03, not bb.startswith("PS"),
                          sum(1 for o in vals if abs(o-np.median(vals))<0.1*np.median(vals))/len(vals)>=0.5,
                          bo > 2.0])
                bets.append({"type":"L","odds":bo,"edge":edge,"score":sc,
                             "won":False,"source":m.get("source","")})

    bdf = pd.DataFrame(bets)
    print(f"   {len(bdf)} signaux ({(bdf['type']=='W').sum()} W, {(bdf['type']=='L').sum()} L)")

    print(f"\n   {'Type':<6} {'Score':>6} {'Paris':>6} {'Win%':>7} {'ROI':>8}")
    print(f"   {'-'*36}")
    for t in ["W","L"]:
        for sc in [1,2,3]:
            sub = bdf[(bdf["type"]==t)&(bdf["score"]>=sc)]
            if len(sub)>=20:
                p = sub.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
                roi = p.sum()/(len(sub)*stake)*100
                ms = " $$$" if roi > 0 else ""
                print(f"   {t:<6} >={sc:>4} {len(sub):>6} {sub['won'].mean():>6.1%} {roi:>+7.1f}%{ms}")

    # Walk-forward Winner
    print(f"\n   Walk-forward Winner score>=2:")
    for src in sorted(bdf["source"].unique()):
        sub = bdf[(bdf["source"]==src)&(bdf["type"]=="W")&(bdf["score"]>=2)]
        if len(sub)>=10:
            p = sub.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
            print(f"     {src}: {len(sub)} paris, ROI {p.sum()/(len(sub)*stake)*100:+.1f}%")

    # Permutation
    sub = bdf[(bdf["type"]=="W")&(bdf["score"]>=2)]
    if len(sub)>=50:
        real = sub.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake,axis=1).sum()/(len(sub)*stake)*100
        odds = sub["odds"].values
        rr = [sum(stake*(o-1) if np.random.random()<1/o else -stake for o in odds)/(len(sub)*stake)*100 for _ in range(1000)]
        pv = np.mean(np.array(rr)>=real)
        v = "SIGNIFICATIF" if pv<0.05 else "non significatif"
        print(f"\n   Permutation: ROI={real:+.1f}%, p={pv:.4f} -> {v}")
        
    print("\n   ATTENTION : DATA LEAKAGE DETECTE")
    print("   B365W = cote du gagnant (retrospectif). Ces resultats sont INVALIDES.")
    print("   Il faudrait des donnees Player1/Player2 pour un vrai backtest.")


def analyze_football():
    print("\n" + "=" * 60)
    print("FOOTBALL ENRICHI")
    print("=" * 60)

    dfs = []
    for rd in [Path("data/raw"), Path("data/raw_other")]:
        if not rd.exists(): continue
        for f in sorted(rd.glob("*.csv")):
            try:
                d = pd.read_csv(f, encoding="latin-1")
                p = f.stem.split("_"); d["league_code"]=p[0]; d["season"]="_".join(p[1:])
                dfs.append(d)
            except: pass
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"HomeTeam":"home_team","AwayTeam":"away_team",
                             "FTHG":"home_goals","FTAG":"away_goals","FTR":"result"})
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date","result"]).sort_values("date").reset_index(drop=True)
    bk_d = [c for c in ["B365D","BWD","IWD","PSD","WHD","VCD"] if c in df.columns]
    for c in bk_d: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Elo
    elos={}; K,HA=25,80; ec=[]
    for _,r in df.iterrows():
        lg=r["league_code"]
        if lg not in elos: elos[lg]={}
        elo=elos[lg]; ht,at=r["home_team"],r["away_team"]
        ec.append(1/(1+abs(elo.get(ht,1500)-elo.get(at,1500))/100))
        hr=elo.get(ht,1500)+HA; ar=elo.get(at,1500)
        e=1/(1+10**((ar-hr)/400))
        s=1.0 if r["result"]=="H" else (0.5 if r["result"]=="D" else 0.0)
        elo[ht]=elo.get(ht,1500)+K*(s-e); elo[at]=elo.get(at,1500)+K*((1-s)-(1-e))
    df["elo_close"]=ec

    # Momentum
    stk={}; mh,ma=[],[]
    for _,r in df.iterrows():
        ht,at=r["home_team"],r["away_team"]
        mh.append(stk.get(ht,0)); ma.append(stk.get(at,0))
        if r["result"]=="H": stk[ht]=stk.get(ht,0)+1; stk[at]=min(0,stk.get(at,0))-1
        elif r["result"]=="A": stk[at]=stk.get(at,0)+1; stk[ht]=min(0,stk.get(ht,0))-1
        else: stk[ht]=0; stk[at]=0
    df["mom_diff"]=np.array(mh)-np.array(ma)

    # Points
    pts={}; ph,pa=[],[]
    for _,r in df.iterrows():
        ht,at=r["home_team"],r["away_team"]
        ph.append(pts.get(ht,0)); pa.append(pts.get(at,0))
        if r["result"]=="H": pts[ht]=pts.get(ht,0)+3
        elif r["result"]=="D": pts[ht]=pts.get(ht,0)+1; pts[at]=pts.get(at,0)+1
        else: pts[at]=pts.get(at,0)+3
    df["pts_diff"]=np.array(ph)-np.array(pa)

    print(f"   {len(df)} matchs")

    # Signaux
    stake=10; sigs=[]
    for _,m in df.iterrows():
        odds={c:m[c] for c in bk_d if pd.notna(m.get(c)) and m.get(c)>1}
        if len(odds)<3: continue
        vals=list(odds.values()); cons=np.mean([1/o for o in vals])
        bb=max(odds,key=odds.get); bo=odds[bb]; edge=cons-1/bo
        if edge<0.01: continue
        sc=sum([edge>=0.03, m["elo_close"]>=0.5,
                sum(1 for o in vals if abs(o-np.median(vals))<0.1*np.median(vals))/len(vals)>=0.6,
                not bb.startswith("PS"), bo>3.0, bb in ["B365D","BWD","WHD","VCD"]])
        bonus=sum([abs(m["mom_diff"])<=2, -10<=m["pts_diff"]<=5,
                   abs(m["mom_diff"])<=1 and abs(m["pts_diff"])<=3])
        sigs.append({"score":sc,"bonus":bonus,"total":sc+bonus,"odds":bo,"edge":edge,
                     "won":m["result"]=="D","season":m.get("season",""),"league":m.get("league_code","")})

    sdf=pd.DataFrame(sigs)
    print(f"   {len(sdf)} signaux (max total={sdf['total'].max()})")

    print(f"\n   {'Filtre':<35} {'Paris':>6} {'Win%':>7} {'ROI':>8}")
    print(f"   {'-'*58}")
    for label,mask in [
        ("Score>=4 (original)", sdf["score"]>=4),
        ("Score>=5 (original)", sdf["score"]>=5),
        ("Total>=6 (enrichi)", sdf["total"]>=6),
        ("Total>=7 (enrichi)", sdf["total"]>=7),
        ("Total>=8 (enrichi)", sdf["total"]>=8),
        ("Score>=4 + mom<=2", (sdf["score"]>=4)&(abs(sdf["score"])<=2)),
        ("Score>=5 + mom<=1", (sdf["score"]>=5)&(abs(sdf["score"])<=1)),
    ]:
        sub=sdf[mask]
        if len(sub)>=30:
            p=sub.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
            roi=p.sum()/(len(sub)*stake)*100
            ms=" $$$" if roi>0 else ""
            print(f"   {label:<35} {len(sub):>6} {sub['won'].mean():>6.1%} {roi:>+7.1f}%{ms}")

    # Walk-forward Total>=7
    print(f"\n   Walk-forward Total>=7:")
    seasons=sorted(sdf["season"].unique())
    tn,tp,pos=0,0,0
    for s in seasons:
        sub=sdf[(sdf["season"]==s)&(sdf["total"]>=7)]
        if len(sub)>=5:
            p=sub.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake, axis=1)
            roi=p.sum()/(len(sub)*stake)*100
            ms=" $" if roi>0 else ""
            print(f"     {s}: {len(sub)} paris, ROI {roi:+.1f}%{ms}")
            tn+=len(sub); tp+=p.sum()
            if roi>0: pos+=1
    if tn>0: print(f"     TOTAL: {tn} paris, ROI {tp/(tn*stake)*100:+.1f}%, {pos}/{len(seasons)} saisons +")

    # Permutation
    best=sdf[sdf["total"]>=7]
    if len(best)>=50:
        real=best.apply(lambda b: stake*(b["odds"]-1) if b["won"] else -stake,axis=1).sum()/(len(best)*stake)*100
        odds=best["odds"].values
        rr=[sum(stake*(o-1) if np.random.random()<1/o else -stake for o in odds)/(len(best)*stake)*100 for _ in range(1000)]
        pv=np.mean(np.array(rr)>=real)
        print(f"\n   Permutation Total>=7: ROI={real:+.1f}%, p={pv:.4f}")
        if pv<0.05: print(f"   >>> SIGNAL SIGNIFICATIF <<<")


def analyze_live():
    key=os.getenv("ODDS_API_KEY","")
    if not key:
        print("\n   Pas de cle API"); return

    print(f"\n{'='*60}")
    print("LIVE MULTI-SPORT")
    print(f"{'='*60}")

    all_picks=[]
    for sk,sn in [("basketball_nba","NBA"),("mma_mixed_martial_arts","UFC"),
                   ("soccer_epl","EPL"),("soccer_spain_la_liga","La Liga"),
                   ("soccer_italy_serie_a","Serie A")]:
        try:
            r=requests.get(f"https://api.the-odds-api.com/v4/sports/{sk}/odds",
                params={"apiKey":key,"regions":"us,uk,eu","oddsFormat":"decimal","markets":"h2h"},timeout=15)
            if r.status_code!=200: continue
            events=r.json()
            picks=[]
            for ev in events:
                h,a=ev["home_team"],ev["away_team"]
                obo={}
                for bk in ev.get("bookmakers",[]):
                    for mkt in bk.get("markets",[]):
                        if mkt["key"]=="h2h":
                            for o in mkt["outcomes"]:
                                if o["name"] not in obo: obo[o["name"]]={}
                                obo[o["name"]][bk["title"]]=o["price"]
                for name,od in obo.items():
                    if len(od)<4: continue
                    vals=list(od.values()); cons=np.mean([1/o for o in vals])
                    bb=max(od,key=od.get); bo=od[bb]; edge=cons-1/bo
                    if edge<0.02: continue
                    sc=sum([edge>=0.03, not any(bb.startswith(s) for s in ["Pinnacle","BetOnline"]),
                            sum(1 for o in vals if abs(o-np.median(vals))<0.1*np.median(vals))/len(vals)>=0.5, bo>2.0])
                    if sc>=3:
                        picks.append({"sport":sn,"match":f"{h} vs {a}","pick":name,
                                      "odds":bo,"edge":edge,"score":sc,"bk":bb})
            if picks:
                print(f"\n   {sn}: {len(picks)} signaux")
                for p in sorted(picks,key=lambda x:x["score"],reverse=True)[:3]:
                    print(f"     [{p['score']}/4] {p['match']} -> {p['pick']} @ {p['odds']:.2f} edge={p['edge']:.1%}")
                all_picks.extend(picks)
        except: pass

    if len(all_picks)>=2:
        print(f"\n   CROSS-SPORT DOUBLES :")
        all_picks.sort(key=lambda x:x["score"],reverse=True)
        seen=set()
        for i in range(len(all_picks)):
            for j in range(i+1,len(all_picks)):
                if all_picks[i]["sport"]!=all_picks[j]["sport"]:
                    k=(all_picks[i]["sport"],all_picks[j]["sport"])
                    if k not in seen:
                        seen.add(k); co=all_picks[i]["odds"]*all_picks[j]["odds"]
                        print(f"     {all_picks[i]['pick']} ({all_picks[i]['sport']}) + "
                              f"{all_picks[j]['pick']} ({all_picks[j]['sport']}) = odds {co:.1f}")
                        if len(seen)>=3: break
            if len(seen)>=3: break


def main():
    print("="*60)
    print("MULTI-SPORT V2")
    print("="*60)
    analyze_tennis()
    analyze_football()
    analyze_live()
    print(f"\n{'='*60}")
    print("TERMINE")
    print("="*60)

if __name__=="__main__": main()
