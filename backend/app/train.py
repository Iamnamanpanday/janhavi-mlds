from __future__ import annotations
import argparse, hashlib, json, subprocess
from datetime import date
from pathlib import Path
import joblib, numpy as np, yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_curve
from xgboost import XGBClassifier
from .config import ARTIFACT_DIR
from .features import FEATURES, GROUPS, constraints

def synthetic(n=1200):
    rng=np.random.default_rng(42); x=rng.normal(size=(n,len(FEATURES))).astype('float32'); y=(1.2*x[:,1]+.8*x[:,8]+.7*x[:,10]- .5*x[:,4]+rng.normal(size=n)>.3).astype(int); return x,y
def tpr_at_fpr(y, s):
    f,t,_=roc_curve(y,s); return float(t[np.where(f<=.001)[0][-1]]) if np.any(f<=.001) else 0.0
def main():
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--ember-dir'); a=p.parse_args()
    # EMBER adapter intentionally requires a preconverted numeric matrix to keep feature schema explicit.
    dataset='synthetic-demo'; x,y=synthetic(300 if a.smoke else 1200)
    ix=np.arange(len(y)); tr,va=ix[ix%5!=0],ix[ix%5==0]
    variants=[('unconstrained',{}),('interaction',{'interaction_constraints':constraints()}),('interaction_monotone',{'interaction_constraints':constraints(),'monotone_constraints':'(0,1,0,1,-1,0,0,0,1,1,1,0)'})]
    rows=[]; best=None
    for name, opts in variants:
      m=XGBClassifier(n_estimators=20 if a.smoke else 100,max_depth=4,learning_rate=.08,n_jobs=1,eval_metric='logloss',**opts).fit(x[tr],y[tr])
      score=m.predict_proba(x[va])[:,1]; row={'name':name,'pr_auc':float(average_precision_score(y[va],score)),'tpr_at_0_1_fpr':tpr_at_fpr(y[va],score)}; rows.append(row)
      if best is None or (row['pr_auc'],row['tpr_at_0_1_fpr'])>(best[0]['pr_auc'],best[0]['tpr_at_0_1_fpr']): best=(row,m)
    raw=best[1].predict_proba(x[va])[:,1]; cal=IsotonicRegression(out_of_bounds='clip').fit(raw,y[va]); f,_,th=roc_curve(y[va],cal.predict(raw)); threshold=float(th[np.where(f<=.001)[0][-1]]) if np.any(f<=.001) else .5
    ARTIFACT_DIR.mkdir(parents=True,exist_ok=True); best[1].save_model(ARTIFACT_DIR/'model.json'); joblib.dump(cal,ARTIFACT_DIR/'calibrator.pkl')
    (ARTIFACT_DIR/'feature_groups.yaml').write_text(yaml.safe_dump({'features':FEATURES,'groups':GROUPS},sort_keys=False))
    card={'model_version':'demo-static-v1','demo_only':True,'dataset':dataset,'date':str(date.today()),'feature_extractor':'static-pe-v1','selection':'PR-AUC then TPR@0.1% FPR','threshold_at_0_1pct_fpr':threshold,'variants':rows,'disclaimer':'Probabilistic static detector; never execute uploads.'}
    (ARTIFACT_DIR/'model_card.json').write_text(json.dumps(card,indent=2)); print(json.dumps(card,indent=2))
if __name__=='__main__': main()
