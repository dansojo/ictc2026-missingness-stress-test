"""Independently aggregate the 943,392 archived prediction records."""
from audit_paths import *
import itertools
import numpy as np
import pandas as pd

cols=["participant_position","participant","family_position","family","primitive_position","sensor_day_id","draw","arm_position","model_position","target_position","condition_position","condition","fit_seq","applicable","replay_status","reference_probability","perturbed_probability","reference_decision","perturbed_decision","absolute_probability_change","decision_flip","instability_burden"]
f=arrow(DOWN/"observed_predictions.arrow",cols)
print("Read",len(f),"prediction records",flush=True)
checks=[]
def check(name,actual,expected,tol=0.0):
    a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
    checks.append({"check":name,"pass":bool(np.allclose(a,b,rtol=0,atol=tol,equal_nan=True)),"max_abs_difference":float(np.nanmax(np.abs(a-b)))})
finite=f.perturbed_probability.notna()&f.applicable
absolute=(f.perturbed_probability-f.reference_probability).abs()
flips=f.perturbed_probability.ge(.5).ne(f.reference_probability.ge(.5))
check("finite absolute changes recomputed from probabilities",absolute[finite],f.loc[finite,"absolute_probability_change"],1e-15)
check("reference decisions at threshold 0.5",f.reference_probability.ge(.5),f.reference_decision)
check("perturbed decisions at threshold 0.5",f.loc[finite,"perturbed_probability"].ge(.5),f.loc[finite,"perturbed_decision"])
check("finite decision reversals",flips[finite],f.loc[finite,"decision_flip"])
burden=np.where(f.applicable,np.where(finite,absolute,1.0),np.nan)
check("abstention-aware burden independently derived",burden,f.instability_burden,1e-15)
f["derived_absolute"]=absolute.where(finite)
f["derived_burden"]=burden
group=["participant_position","family_position","arm_position","model_position","target_position"]
condition_group=f.groupby(group+["condition"],sort=True).derived_absolute.median().rename("value").reset_index()
participant_drift=condition_group.groupby(["participant_position","condition"],sort=True).value.median().reset_index()
global_drift=participant_drift.groupby("condition",sort=True).value.median()
old_groups=parquet(DOWN/"statistics/condition_group_descriptives.parquet")
comp=condition_group.merge(old_groups,on=group+["condition"],validate="one_to_one")
check("1920 condition/group finite drift medians",comp.value,comp.finite_abs_change_median,1e-15)
summary=[]
for condition in ["scattered_random_20pct","contiguous_20pct","event_boundary_20pct"]:
    selected=f.condition.eq(condition);valid=selected&finite
    expected=int(selected.sum());applicable=int((selected&f.applicable).sum());nf=int(valid.sum());nflip=int(flips[valid].sum())
    summary.append({"condition":condition,"expected":expected,"applicable":applicable,"unavailable":expected-applicable,"finite":nf,"abstained":applicable-nf,"flips":nflip,"reversal_percent":100*nflip/applicable,"participant_first_median_abs_change":float(global_drift[condition]),"pooled_median_abs_change":float(absolute[valid].median())})

keys=group+["primitive_position","sensor_day_id","draw"]
wide=f.loc[f.condition.isin(["scattered_random_20pct","contiguous_20pct"])].pivot(index=keys,columns="condition",values="derived_burden")
delta=(wide.contiguous_20pct-wide.scattered_random_20pct).rename("delta").reset_index()
group_delta=delta.groupby(group,sort=True).delta.median().reset_index()
participant=group_delta.groupby("participant_position",sort=True).delta.median()
old_participant=parquet(DOWN/"statistics/participant_summaries.parquet").sort_values("participant_position")
check("10 participant corroboration medians from prediction rows",participant,old_participant.p_median,1e-15)
old_group=parquet(DOWN/"statistics/group_cells.parquet").sort_values(group)
check("640 group corroboration medians",group_delta.delta,old_group.g_median,1e-15)
values=participant.to_numpy(float)
idx=np.random.Generator(np.random.PCG64(42)).integers(0,10,size=(10000,10),dtype=np.int64,endpoint=False)
boot=np.median(values[idx],axis=1)
ci=np.quantile(boot,[.025,.975],method="linear")
signs=np.asarray(list(itertools.product([-1.,1.],repeat=10)))
observed=float(np.median(values));p=float((np.abs(np.median(signs*values,axis=1))>=abs(observed)).mean())
decision={"participant_median":observed,"ci_low":float(ci[0]),"ci_high":float(ci[1]),"positive_participants":int((values>0).sum()),"sign_p_value":p,"participants":len(values),"groups_per_participant":sorted(group_delta.groupby("participant_position").size().unique().tolist()),"primary_keys":len(wide)}
fit=arrow(DOWN/"fit/fit_ledger.arrow",["fit_seq","heldout_position","arm_position","model_position","target_position"])
expected_fit=(((f.participant_position*2+f.arm_position)*2+f.model_position)*4+f.target_position)
check("fit identities associated with each prediction",f.fit_seq,expected_fit)
fit_info={"ledger_rows":len(fit),"unique_fit_ids":int(fit.fit_seq.nunique()),"unique_prediction_fit_ids":int(f.fit_seq.nunique()),"conditions_per_fit":sorted(f.groupby("fit_seq").condition.nunique().unique().tolist())}

paper=[]
def claim(name,actual,reported,decimals=None):
    text=f"{actual:.{decimals}f}" if decimals is not None else str(int(actual))
    want=f"{reported:.{decimals}f}" if decimals is not None else str(int(reported))
    paper.append({"location":"III-B / IV-B / Figure 2c","claim":name,"reported":want,"recomputed":float(actual),"match_at_printed_precision":text==want})
claim("total prediction records",len(f),943392)
claim("fit identities",len(fit),160)
for row,drift,nflip,percent,applicable in zip(summary,[.000142,.001352,.004128],[4384,7747,8636],[1.394,2.464,2.818],[314464,314464,306464]):
    for name,actual,reported,decimals in [("expected",row["expected"],314464,None),("applicable",row["applicable"],applicable,None),("flips",row["flips"],nflip,None),("reversal percent",row["reversal_percent"],percent,3),("finite drift",row["participant_first_median_abs_change"],drift,6)]:claim(row["condition"]+" "+name,actual,reported,decimals)
for name,reported in [("finite",305360),("abstained",1104),("unavailable",8000)]:claim("boundary "+name,summary[2][name],reported)
for name,reported in [("participant_median",0),("ci_low",0),("ci_high",0),("positive_participants",1),("sign_p_value",1)]:claim(name,decision[name],reported,6)
claim("groups per participant",decision["groups_per_participant"][0],64)
save_json("downstream_recomputation.json",{"method":"independent probability arithmetic and hierarchical aggregation from archived prediction rows; models not refitted or rescored","rows":len(f),"summary":summary,"corroboration":decision,"fit_topology":fit_info,"consistency_checks":checks,"paper_claims":paper})
print(json.dumps({"summary":summary,"corroboration":decision,"fit_topology":fit_info,"failed_checks":[x for x in checks if not x["pass"]],"paper_checks":len(paper),"paper_mismatches":[x for x in paper if not x["match_at_printed_precision"]]},ensure_ascii=False))
