"""Fixed, participant-disjoint prediction and matched-record stress semantics."""
import hashlib
import json
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT_SEED = 20260911
FEATURE_NAMES = {
    'StudentLife': ['active_fraction', 'mean_activity_code'],
    'ExtraSensory': ['magnitude_mean','magnitude_std','magnitude_median','magnitude_p90'],
}

def digest_object(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def seed_for(*identity):
    return int.from_bytes(bytes.fromhex(digest_object([ROOT_SEED,*identity]))[:8],'big')

def features(dataset, values):
    v=np.asarray(values,dtype=np.float64)
    if v.ndim!=1 or len(v)<2 or not np.isfinite(v).all():
        raise ValueError('Features need at least two finite scalar observations')
    if dataset=='StudentLife':
        if not np.isin(v,[0.,1.,2.]).all():raise ValueError('Unknown activity code is not a known state')
        return np.array([np.mean(v>0),np.mean(v)],dtype=np.float64)
    if dataset=='ExtraSensory':
        if (v<0).any():raise ValueError('Acceleration magnitude cannot be negative')
        return np.array([np.mean(v),np.std(v,ddof=0),np.median(v),np.quantile(v,.9,method='linear')])
    raise ValueError('Unknown dataset')

def mask_pair(n, identity):
    if n<5:raise ValueError('At least five records are needed for 20 percent deletion')
    k=int(np.floor(.2*n))
    rng=np.random.Generator(np.random.PCG64(seed_for(*identity,'scattered_random_20pct')))
    scattered=np.sort(rng.choice(n,k,replace=False))
    rng=np.random.Generator(np.random.PCG64(seed_for(*identity,'contiguous_20pct')))
    start=int(rng.integers(0,n-k+1))
    return scattered,np.arange(start,start+k,dtype=np.int64)

def participant_weights(groups):
    g=np.asarray(groups)
    names,counts=np.unique(g,return_counts=True)
    if not len(names):raise ValueError('No training participants')
    weights={p:len(g)/(len(names)*n) for p,n in zip(names,counts)}
    return np.array([weights[p] for p in g],dtype=np.float64)

def fingerprint(fold):
    scaler=fold['scaler'];model=fold['model']
    return digest_object({'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
        'var':scaler.var_.tolist(),'coef':model.coef_.tolist(),
        'intercept':model.intercept_.tolist(),'classes':model.classes_.tolist(),
        'prior':float(fold['prior'])})

def fit_fold(x,y,groups,heldout):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.int64);g=np.asarray(groups)
    if len(x)!=len(y) or len(x)!=len(g) or not np.isfinite(x).all():
        raise ValueError('Misaligned or nonfinite unmasked training features')
    train=np.flatnonzero(g!=heldout);test=np.flatnonzero(g==heldout)
    if not len(test) or not len(train):raise ValueError('Empty participant-disjoint split')
    if set(y[train])!={0,1}:raise ValueError('Outer training set lacks either label class')
    weights=participant_weights(g[train])
    scaler=StandardScaler().fit(x[train],sample_weight=weights)
    model=LogisticRegression(C=1.,solver='lbfgs',l1_ratio=0.,max_iter=2000,tol=1e-8,
                             random_state=ROOT_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(scaler.transform(x[train]),y[train],sample_weight=weights)
    fold={'model':model,'scaler':scaler,'prior':float(np.average(y[train],weights=weights)),
          'train_indices':train,'test_indices':test,
          'train_participants':sorted(np.unique(g[train]).tolist()),
          'heldout_participant':str(heldout),'n_train':len(train),'n_test':len(test),
          'n_iterations':int(model.n_iter_[0]),
          'weighted_train_prevalence':float(np.average(y[train],weights=weights)),
          'weight_total_by_participant':{str(p):float(weights[g[train]==p].sum()) for p in np.unique(g[train])}}
    fold['fingerprint']=fingerprint(fold)
    return fold

def predict(fold,x):
    return fold['model'].predict_proba(fold['scaler'].transform(np.asarray(x,dtype=np.float64)))[:,1]

def metrics(y,p):
    y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.float64)
    if len(y)!=len(p) or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid targets or probabilities')
    eps=np.finfo(np.float64).eps;q=np.clip(p,eps,1-eps)
    both=np.unique(y).size==2;decision=p>=.5
    ba=float(.5*(np.mean(decision[y==1])+np.mean(~decision[y==0]))) if both else float('nan')
    return {'log_loss':float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
            'brier':float(np.mean((p-y)**2)),
            'auc':float(roc_auc_score(y,p)) if both else float('nan'),
            'balanced_accuracy':ba,'accuracy':float(np.mean(decision==y)),
            'probability_sd':float(np.std(p,ddof=0)),
            'predicted_positive_fraction':float(np.mean(decision))}

def change_metrics(y,observed,perturbed):
    observed=np.asarray(observed,dtype=np.float64);perturbed=np.asarray(perturbed,dtype=np.float64)
    m=metrics(y,perturbed)
    m.update(drift=float(np.mean(np.abs(perturbed-observed))),
             flip_rate=float(np.mean((perturbed>=.5)!=(observed>=.5))))
    return m

def bootstrap_median(values,identity,draws=10000):
    values=np.asarray(values,dtype=np.float64);values=values[np.isfinite(values)]
    if not len(values):return [float('nan')]*3
    rng=np.random.Generator(np.random.PCG64(seed_for('bootstrap',identity)))
    boot=np.median(values[rng.integers(0,len(values),size=(draws,len(values)))],axis=1)
    low,high=np.quantile(boot,[.025,.975])
    return [float(np.median(values)),float(low),float(high)]
