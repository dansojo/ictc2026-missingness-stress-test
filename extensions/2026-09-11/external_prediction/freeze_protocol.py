"""One-time freeze after data-only feasibility checks and before external fits."""
from pathlib import Path
from datetime import datetime, timezone
import json
import platform
import numpy, pandas, sklearn, scipy, pyarrow
from run_prediction import sha, save_json, validate_windows
import prediction_core as c

HERE=Path(__file__).resolve().parent
INVENTORY=HERE.parent/'external_inventory'
assert not (HERE/'FROZEN_PROTOCOL.json').exists(), 'Never overwrite an execution freeze'
verified=json.loads((INVENTORY/'DEDUPLICATED_VERIFICATION.json').read_text(encoding='utf8'))
assert verified['status']=='PASS' and verified['model_fits']==0
input_file=INVENTORY/'eligible_prediction_windows_deduplicated.parquet'
assert sha(input_file)==verified['eligible_prediction_windows_sha256']
windows=pandas.read_parquet(input_file)
denominators=validate_windows(windows)
assert denominators.groupby('dataset').size().to_dict()=={'ExtraSensory':53,'StudentLife':49}
assert windows.groupby('dataset').size().to_dict()=={'ExtraSensory':5085,'StudentLife':1126}
denominators.to_csv(HERE/'FROZEN_PARTICIPANT_DENOMINATORS.csv',index=False)
final_text=(HERE/'PROTOCOL_DRAFT.md').read_text(encoding='utf8')
final_text=final_text.replace('Status: data-only feasibility inventory underway; no external predictive performance has been inspected. The final eligible input table and this design will be hash-frozen before model fitting.',
    'Status: frozen after data-only feasibility and before any external-data model fitting. Synthetic unit tests were run; no external predictive performance has been inspected. This protocol and the final eligible input table are hash-pinned in FROZEN_PROTOCOL.json.')
final_text+='''
## Final data-only design refinement and eligible denominator

Before fitting, the duplicate audit found15,086 repeated-timestamp groups in selected StudentLife windows, including947groups with conflicting activity codes. The final deterministic cleanup collapses an agreeing timestamp to one observation and discards every observation at a conflicting timestamp. It removes15,070 identical extra rows and1,915 conflicting rows; the same >=60record/>=12hspan eligibility is then checked. No evaluated window or participant is lost. Prior arrays are preserved. This is new prediction preprocessing; the old published feature analysis is not recomputed or relabeled as having used this cleanup. Timestamps in the final predictor arrays must strictly increase. This policy was chosen from source-quality evidence before predictive results, not by comparing model scores.

Final StudentLife:49participants,1,126prior-day windows,249poor/877good; per-person1–58days(median21). Eight people have one targetclass only, so AUROC/balanced accuracy use41people while losses/drift use49. Final ExtraSensory:53participants,5,085nonoverlapping windows,1,457sleeping/3,628other; per-person20–161windows(median100). One person has one targetclass only, so AUROC/balanced accuracy use52people while losses/drift use53. All outer training sets contain both classes. StudentLife's previous-day alignment is a coarse association; original EMA response times can be almost24h after that day's end and travel timezones are unavailable. Local release IDs, rather than the original paper's participant count, define these denominators.

Earlier inventory variants(30/60min, hourly anchors, alternative support counts/response-hour filters) were data-only feasibility probes. They were not fitted or compared for predictive performance. The final two tasks and settings above are the only external predictive experiments in this revision. The20s acquisition guard and StudentLife canonicalization are recorded pre-fit refinements; no score has guided them.
'''
(HERE/'PROTOCOL_FINAL.md').write_text(final_text,encoding='utf8')
sources=['prediction_core.py','run_prediction.py','test_prediction_core.py','test_window_contracts.py',
         'PROTOCOL_FINAL.md','freeze_protocol.py']
protocol={'status':'frozen_before_external_data_models','frozen_utc':datetime.now(timezone.utc).isoformat(),
    'scope':'post_review_exploratory','external_data_model_fits_before_freeze':0,
    'synthetic_unit_tests_performed':True,
    'windows_path':str(input_file),'windows_sha256':sha(input_file),
    'inventory_verification_sha256':sha(INVENTORY/'DEDUPLICATED_VERIFICATION.json'),
    'canonicalization_sha256':sha(INVENTORY/'CANONICALIZATION.json'),
    'sources':{name:sha(HERE/name) for name in sources},
    'config':{'seed':20260911,'draws':50,'deletion_fraction':.2,
        'datasets':['ExtraSensory','StudentLife'],'splits':'leave_one_participant_out',
        'weighting':'equal_total_training_weight_per_participant_normalized_mean1',
        'features':c.FEATURE_NAMES,'estimator':'L2 LogisticRegression','C':1.0,'l1_ratio':0.0,
        'solver':'lbfgs','max_iter':2000,'tol':1e-8,'decision_threshold':.5,
        'bootstrap_draws':10000,'near_constant_probability_sd_flag':.01,
        'sample_exclusion_based_on_performance':False,'hyperparameter_search':False},
    'environment':{'python':platform.python_version(),'platform':platform.platform(),
        'numpy':numpy.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__,
        'scipy':scipy.__version__,'pyarrow':pyarrow.__version__}}
save_json(HERE/'FROZEN_PROTOCOL.json',protocol)
print(json.dumps({'status':protocol['status'],'sha256':sha(HERE/'FROZEN_PROTOCOL.json'),
                  'windows':len(windows),'planned_model_fits':len(denominators)}))
