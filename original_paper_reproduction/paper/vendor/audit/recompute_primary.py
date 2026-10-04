"""Independently derive manuscript statistics from the 80,000 primary replay rows."""
from audit_paths import *
import itertools
import numpy as np
import pandas as pd

checks = []
def check(name, actual, expected, tol=0.0):
    ok = bool(np.allclose(actual, expected, rtol=0, atol=tol, equal_nan=True))
    checks.append({"check": name, "pass": ok, "max_abs_difference": float(np.nanmax(np.abs(np.asarray(actual)-np.asarray(expected))))})

frame = parquet(DIAG/"primary_comparison.parquet")
g2root=SDD/"task-4c2-v2-official"
g2manifest=read_json(g2root/"manifest.json")
assert digest(g2root/"manifest.json")=="223b7b55e66d1dc8ecd522cde77c992934bde0897fbf9b7745b8bf18a361a7d1"
assert digest(g2root/"cell_replays.parquet")==g2manifest["official_artifacts"]["cell_replays"]["sha256"]
g2=parquet(g2root/"cell_replays.parquet")
g2keys=["subject_id","sensor_day_id","primitive","draw","scenario"]
inherited=frame.scenario.eq("contiguous_20pct")
missing_detail_count=int(frame.loc[inherited,"reference_value"].isna().sum())
details=g2[g2.scenario.eq("contiguous_20pct")][g2keys+["original_value","masked_value","standardizer","standardized_absolute_error"]].rename(columns={"original_value":"g2_reference","masked_value":"g2_masked","standardizer":"g2_scale","standardized_absolute_error":"g2_error"})
frame=frame.merge(details,on=g2keys,how="left",validate="one_to_one")
inherited=frame.scenario.eq("contiguous_20pct")
check("40000 inherited errors match original G2 record keys",frame.loc[inherited,"standardized_absolute_error"],frame.loc[inherited,"g2_error"],1e-12)
for target,source in [("reference_value","g2_reference"),("masked_value","g2_masked"),("standardizer","g2_scale")]:frame.loc[inherited,target]=frame.loc[inherited,source]
finite = frame.status.eq("finite")
errors = (frame.masked_value-frame.reference_value).abs()/frame.standardizer
check("standardized errors from stored feature values", errors[finite], frame.loc[finite,"standardized_absolute_error"], 1e-12)
keys = ["feature_family","subject_id","sensor_day_id","primitive","draw"]
c = frame[frame.scenario.eq("contiguous_20pct")].set_index(keys).sort_index()
r = frame[frame.scenario.eq("scattered_random_20pct")].set_index(keys).sort_index()
assert c.index.equals(r.index) and c.index.is_unique
valid = c.status.eq("finite") & r.status.eq("finite")
valid &= np.isfinite(c.standardized_absolute_error) & np.isfinite(r.standardized_absolute_error)
pairs = pd.DataFrame({"finite":valid, "delta":np.where(valid,c.standardized_absolute_error-r.standardized_absolute_error,np.nan)},index=c.index).reset_index()
participant = pairs.groupby(["feature_family","subject_id"],sort=True).delta.median().dropna().reset_index()
old_participant = parquet(DIAG/"participant_deltas.parquet").sort_values(["feature_family","subject_id"])
check("40 independently aggregated participant deltas",participant.delta,old_participant.delta,1e-12)

families=["event_count","state_ratio","intensity","timing"]
results=[]
for family in families:
    values=participant.loc[participant.feature_family.eq(family),"delta"].to_numpy(float)
    signs=np.asarray(list(itertools.product([-1.0,1.0],repeat=len(values))))
    observed=abs(values.mean())
    permutations=np.abs((signs*values).mean(axis=1))
    p=float((permutations>=observed).mean())
    payload=json.dumps(["participant-cluster-bootstrap-v1",42,family],ensure_ascii=False,separators=(",",":")).encode()
    seed=int.from_bytes(hashlib.sha256(payload).digest()[:8],"big")
    idx=np.random.Generator(np.random.PCG64(seed)).integers(0,len(values),size=(10000,len(values)))
    lo,hi=np.quantile(np.median(values[idx],axis=1),[.025,.975],method="linear")
    family_pairs=pairs[pairs.feature_family.eq(family)]
    results.append({"feature_family":family,"participants":len(values),"paired_total":len(family_pairs),"paired_finite":int(family_pairs.finite.sum()),"participant_median":float(np.median(values)),"participant_mean":float(np.mean(values)),"ci_low":float(lo),"ci_high":float(hi),"positive_count":int((values>0).sum()),"raw_p_value":p})
order=sorted(range(4),key=lambda i:results[i]["raw_p_value"])
running=0.0
for rank,index in enumerate(order):
    running=max(running,min(1.0,(4-rank)*results[index]["raw_p_value"]))
    results[index]["holm_p_value"]=running
stored=parquet(DIAG/"family_inference.parquet").set_index("feature_family")
for row in results:
    for col in ["participant_median","participant_mean","ci_low","ci_high","raw_p_value","holm_p_value"]:
        check(row["feature_family"]+" "+col,row[col],float(stored.loc[row["feature_family"],col]),1e-12)
    check(row["feature_family"]+" positive participants",row["positive_count"],int(stored.loc[row["feature_family"],"direction_agreement_count"]))

family_map=dict(frame[["primitive","feature_family"]].drop_duplicates().itertuples(index=False,name=None))
dose_replays=g2[g2.scenario.isin(["contiguous_10pct","contiguous_20pct","contiguous_40pct"])].copy()
dose_replays["feature_family"]=dose_replays.primitive.map(family_map)
dose_replays["missingness_rate"]=dose_replays.scenario.map({"contiguous_10pct":.1,"contiguous_20pct":.2,"contiguous_40pct":.4})
dose_replays["derived_error"]=(dose_replays.masked_value-dose_replays.original_value).abs()/dose_replays.standardizer
check("120000 inherited dose errors from original feature values",dose_replays.derived_error,dose_replays.standardized_absolute_error,1e-12)
dose_part=dose_replays.groupby(["subject_id","feature_family","missingness_rate"],sort=True).derived_error.median().rename("participant_median").reset_index()
stored_dose_part=parquet(DIAG/"dose_response_participant.parquet").sort_values(["subject_id","feature_family","missingness_rate"])
check("120 participant dose estimates from original 120000 replay rows",dose_part.participant_median,stored_dose_part.participant_median,1e-12)
dose=dose_part.groupby(["feature_family","missingness_rate"],sort=True).participant_median.median().reset_index()
old_dose=parquet(DIAG/"dose_response_family.parquet").sort_values(["feature_family","missingness_rate"])
check("12 dose estimates recomputed from 120 participant dose estimates",dose.participant_median,old_dose.participant_median,1e-12)
mask=parquet(DIAG/"mask_ledger.parquet")
deletions=parquet(DIAG/"deletion_audit.parquet")
audit=deletions.merge(mask[g2keys+["mask_status"]],on=g2keys,validate="one_to_one")
random_audit=audit[audit.scenario.eq("scattered_random_20pct")]
matching={"random_rows":int(mask.scenario.eq("scattered_random_20pct").sum()),"boundary_rows":int(mask.scenario.eq("event_boundary_20pct").sum()),"random_ineligible_rows":int(random_audit.mask_status.ne("eligible").sum()),"eligible_random_exact_match_failures":int((~random_audit.loc[random_audit.mask_status.eq("eligible"),"exact_match"]).sum())}
selected=c.reset_index().drop_duplicates(["subject_id","sensor_day_id","primitive"])
support={"participants":int(selected.subject_id.nunique()),"primitives":int(selected.primitive.nunique()),"cells":len(selected),"days_per_participant_primitive":sorted(selected.groupby(["subject_id","primitive"]).size().unique().tolist()),"draws":int(c.reset_index().draw.nunique()),"new_mask_rows":len(mask),**matching}

paper=[]
def paper_claim(location,name,actual,printed,decimals=None):
    actual_text=f"{actual:.{decimals}f}" if decimals is not None else str(int(actual))
    expected_text=f"{printed:.{decimals}f}" if decimals is not None else str(int(printed))
    paper.append({"location":location,"claim":name,"reported":expected_text,"recomputed":float(actual),"match_at_printed_precision":actual_text==expected_text})
expected={"event_count":(4995,5000,0.,-.0209032669,.0475907528,3,.46875,.9375),"state_ratio":(10000,10000,.463,.284,.555,10,.00195,.0078),"intensity":(14840,15000,.376,.261,.504,10,.00195,.0078),"timing":(9790,10000,0.,0.,.0073314213,2,.5,.9375)}
for row in results:
    family=row["feature_family"];e=expected[family]
    paper_claim("Table I / Results",family+" finite pairs",row["paired_finite"],e[0])
    paper_claim("Table I / Results",family+" scheduled pairs",row["paired_total"],e[1])
    paper_claim("Abstract / IV-A / Figure 2a",family+" median",row["participant_median"],e[2],3)
    paper_claim("IV-A",family+" raw p",row["raw_p_value"],e[6],5)
    paper_claim("IV-A",family+" Holm p",row["holm_p_value"],e[7],4)
    if family in ["state_ratio","intensity"]:
        paper_claim("Abstract / IV-A",family+" CI lower",row["ci_low"],e[3],3)
        paper_claim("Abstract / IV-A",family+" CI upper",row["ci_high"],e[4],3)
        paper_claim("IV-A",family+" positive participants",row["positive_count"],e[5])
for family,want in zip(families,[1.097,.801,1.618,.319]):
    value=dose.loc[dose.feature_family.eq(family)&dose.missingness_rate.eq(.4),"participant_median"].iloc[0]
    paper_claim("IV-A / Figure 2b",family+" 40% dose",value,want,3)
for key,want in [("participants",10),("primitives",8),("cells",800),("draws",50),("new_mask_rows",80000)]:paper_claim("III-A",key,support[key],want)

save_json("primary_recomputation.json",{"method":"independent arithmetic over stored per-mask feature values, including original G2 inherited replay details; no raw sensor remasking","inherited_detail_recovery":{"primary_rows_missing_feature_details":missing_detail_count,"recovered_from":str(g2root/"cell_replays.parquet"),"original_g2_replay_rows":len(g2),"used_dose_replay_rows":len(dose_replays)},"support":support,"results":results,"dose_from_participants":dose.to_dict(orient="records"),"consistency_checks":checks,"paper_claims":paper})
print(json.dumps({"support":support,"results":results,"failed_consistency_checks":[x for x in checks if not x["pass"]],"paper_checks":len(paper),"paper_mismatches":[x for x in paper if not x["match_at_printed_precision"]]},ensure_ascii=False))
