"""Author-facing tables from completed results and an independently passed audit."""
from pathlib import Path
import json
import pandas as pd
HERE=Path(__file__).resolve().parent
METHODS=['CD_ALL_DAY','CD_TIME_MEAN','CD_TIME_MEDIAN','CD_WEEKTYPE_MEAN','CD_WEEKTYPE_MEDIAN','CD_PAST_TIME_MEAN','CD_PAST_RECENT_MEAN','CD_SIMILAR_COPY']
LABELS=['다른 날 전체 평균','동일 시간 평균','동일 시간 중앙값','평일/주말 평균','평일/주말 중앙값','과거 동일 시간 평균','최근 날짜 가중 평균','비슷한 날 복사']
PRIMITIVES=['screen_load_24h','phone_activity_load_24h','usage_load_24h','mobile_light_exposure_24h','wearable_light_exposure_24h']
def read(name):return pd.read_csv(HERE/'summary_01'/name,float_precision='round_trip')
def run():
    destination=HERE/'AUTHOR_REPORT.md'
    if destination.exists():raise ValueError('Preserve previous author report')
    audit=json.loads((HERE/'independent_audit_02/verification.json').read_text(encoding='utf8'))
    if audit['status']!='passed':raise ValueError('Independent audit must pass before completion report')
    mf=json.loads((HERE/'full_01/manifest.json').read_text(encoding='utf8'))
    freeze=json.loads((HERE/'FROZEN_PROTOCOL.json').read_text(encoding='utf8'))
    summary=read('summary.csv');fallback=read('fallback_totals.csv')
    lines=['# 같은 참가자의 다른 날짜를 이용한 대치: 저자 검토 보고서','',
        (HERE/'FINDINGS_KO.md').read_text(encoding='utf8'),'',
        '## 실제 실행 및 보존','',
        f'- 동결 시각: {freeze["frozen_utc"]}. 실제 donor 점검과 코드·설정 동결 후 신규 결과를 계산했다.',
        f'- 제공된 참가자 전원 10명, 5개 특징, 기존 평가 날짜 500개 셀, 50개 draw, 두 결측 구조에서 신규 8개 방법 {mf["rows"]:,}행을 실행했다.',
        f'- 신규 결과 상태: 계산 가능 {mf["status_counts"].get("finite",0):,}행, 적용 불가 {mf["status_counts"].get("not_applicable",0):,}행, 구조상 계산 불가 {mf["status_counts"].get("structurally_unavailable",0):,}행. 기타 상태도 full_01/manifest.json 및 상태 표에 남아 있다.',
        '- 기존 500,000행은 원래 파일 그대로 보존했다. 참가자·평가 날짜·마스크·reference·scale 및 기존 revision4 파일의 해시는 실행·검증 전후 대조했다. 기존 원고 및 최신본 포인터는 변경하지 않았고 업로드하지 않았다.',
        '- donor 원자료는 기존 853일 × 5개 특징 = 4,265개 셀이다. 500개 평가 셀의 원자료·유효성·시간·달력·출처 식별자는 캐시 저장 전후 모두 원본과 정확히 일치했다. 자세한 수와 날짜별 목록은 donor_inventory/FEASIBILITY.md에 있다.','',
        '## 결과를 읽는 방법','',
        '아래 수치는 **비교군의 표준화 절대 특징 오차 − 해당 대치 방법의 오차**다. 양수는 개선, 음수는 악화다. 정확도나 개선 백분율이 아니다. 원래 reference와 scale을 유지했다. 참가자마다 날짜·draw의 차이의 중앙값을 구하고, 그 10개 값의 중앙값을 표시했다.',
        '모든 비교는 같은 원행에서 차이를 계산한 뒤 요약한다. 중앙값은 가법적이지 않으므로, 두 방법의 M0 대비 중앙값을 단순히 빼서 방법 간 개선량으로 해석하면 안 된다. 예를 들어 화면의 동일 시간 평균은 M0 대비−0.3192, 다른 날 평균은−0.4289지만, 동일 시간 평균 대 다른 날 평균의 짝지은 개선 중앙값은0이다. 아래 별도 비교군 표를 그대로 사용한다.',
        '동일 조건에서 계산 가능한 모든 적용 방법의 교집합을 사용했다. 수치형 특징은 기존·신규 18개 방법, 이진 상태는 수치형 전용 네 방법을 제외한 14개 방법이다. 반복 마스크가 많아도 독립 참가자는 10명이다. 모든 참가자별 값과 양수/0/음수 개수, 95% 참가자 bootstrap 구간은 summary_01/summary.csv와 participant_effects.csv에 있다.',
        '표의 열 순서는 화면 상태·활동 상태·사용량·모바일 조도·웨어러블 조도다. 서로 다른 특징을 하나의 intensity 수치로 합치지 않았다. N/A는 이진 상태에 적용하지 않은 중앙값 방법이다.','']
    for comparator,clabel in [('M0','대치하지 않은 경우'),('CD_ALL_DAY','다른 날 전체 평균')]:
        for geometry,glabel in [('contiguous_20pct','연속 결측'),('scattered_random_20pct','무작위 분산 결측')]:
            lines += [f'### {glabel}: {clabel} 대비','',
                '| 방법 | 화면 | 활동 | 사용량 | 모바일 조도 | 웨어러블 조도 |',
                '|---|---:|---:|---:|---:|---:|']
            for method,label in zip(METHODS,LABELS):
                values=[]
                for primitive in PRIMITIVES:
                    r=summary.loc[summary.comparator.eq(comparator)&summary.geometry.eq(geometry)&summary.method.eq(method)&summary.primitive.eq(primitive)].iloc[0]
                    values.append(f'{r.median_gain:+.4f}' if r.applicable else 'N/A')
                lines.append('| '+label+' | '+' | '.join(values)+' |')
            lines+=['']
    lines+=['## Fallback 의존성','',
        '아래는 실제 대치를 시도한 삭제 유효 관측 중 fallback으로 채운 비율이다. 반복 마스크에 등장한 채움 횟수의 비율이며, 고유 원자료 관측 수의 비율이 아니다. 적용 불가 사례는 실제 시도 분모에서 제외하고 별도로 보고한다.','']
    for geometry,glabel in [('contiguous_20pct','연속 결측'),('scattered_random_20pct','무작위 분산 결측')]:
        lines += [f'### {glabel} fallback 비율','',
            '| 방법 | 화면 | 활동 | 사용량 | 모바일 조도 | 웨어러블 조도 |',
            '|---|---:|---:|---:|---:|---:|']
        for method,label in zip(METHODS,LABELS):
            values=[]
            for primitive in PRIMITIVES:
                rows=fallback.loc[fallback.geometry.eq(geometry)&fallback.method.eq(method)&fallback.primitive.eq(primitive)]
                denominator=float(rows.attempted_n.sum());numerator=float(rows.fallback_n.sum())
                values.append(f'{100*numerator/denominator:.2f}%' if denominator else 'N/A')
            lines.append('| '+label+' | '+' | '.join(values)+' |')
        lines+=['']
    lines+=['동일 시간 fallback, 다른 날 전체 평균 fallback, 당일 남은 값 fallback은 fallback_totals.csv와 fallback_by_person.csv에 구분했다. 비슷한 날을 선택하지 못한 경우와 선택했지만 대응하는 관측이 없는 경우도 별도 열과 similarity_donor_ledger.csv에 기록했다.','',
        '## 독립 검증 근거','',
        f'- 신규 전체 {audit["new_rows"]:,}행의 키·상속 메타데이터·오차 계산·상태·채움 횟수를 검사했다. 기존 결과 {audit["archived_rows"]:,}행의 원본 해시도 유지됐다.',
        f'- 모든 500개 평가 셀과 두 결측 구조에서 사전에 정한 draw0의 {audit["independent_raw_method_rows"]:,}개 방법 결과, 실제 raw 채움 {audit["independent_raw_values"]:,}개를 별도 구현으로 다시 계산했다.',
        f'- raw 값 {audit["independent_raw_exact_values"]:,}개는 정확히 같았다. 전체 raw 최대 절대 차이는 {audit["independent_raw_max_absdiff"]:.12g}, 특징값 최대 절대 차이는 {audit["independent_feature_max_absdiff"]:.12g}였다. 수치형은 사전 설정 rtol=atol=1e-12로 검사했고, 이진 상태·선택 날짜·개수는 정확 비교했다.',
        f'- 동일한 고정 draw0 범위의 공개 특징 계산 API 검사 {audit["public_checks"]:,}개가 정확히 일치했다. 독립 구현은 실행용 대치·요약 함수를 호출하지 않았다.',
        '- 모든 효과·참가자별 값·bootstrap 구간·fallback/상태/선택 donor 요약을 원래 결과 행에서 독립적으로 다시 계산했다. 구체적인 항목별 수와 파일 해시는 independent_audit_02/verification.json에 있다.',
        '- 첫 독립 검사에서는 id07 화면 상태의 과거 평균에서0/1 불일치가 발견됐다. 정확한 평균은1/2이고 실험 코드는 동률 규칙대로0을 채웠지만, 첫 검증기의 부동소수점 평균은0.5000000000000001이 되어1로 판단했다. 첫 실패 기록과 원인 진단을 independent_audit_01에 보존했다. 실험 결과를 덮어쓰지 않고 검증기를 별도 버전으로 수정한 후 전체 검사를 다시 수행했다. 수정 이유와 정확 비교 증거는 검증 보완 기록에 남겼다.',
        '- 독립 raw 재계산은 draw0만 수행했다. draw1–49까지 raw 값을 모두 다른 구현으로 재계산했다고 주장하지 않는다. 모든 draw의 행·메타데이터·통계 검사는 수행했다.','',
        '아래는 특징별 독립 계산 차이다. 서로 단위가 다른 raw 최대 차이를 하나의 크기 비교로 해석하지 않는다.','',
        '| 특징 | raw 단위 | 대조한 채움 수 | 정확 일치 수 | 최대 raw 절대 차이 | 최대 특징 절대 차이 |',
        '|---|---|---:|---:|---:|---:|']
    raw_checks=pd.read_csv(HERE/'independent_audit_02/independent_raw_checks.csv',float_precision='round_trip')
    for primitive,label,unit in zip(PRIMITIVES,['화면','활동','사용량','모바일 조도','웨어러블 조도'],['0/1','0/1','ms','lux','lux']):
        z=raw_checks.loc[raw_checks.primitive.eq(primitive)]
        lines.append(f'| {label} | {unit} | {int(z.audited_points.sum()):,} | {int(z.exact_points.sum()):,} | {z.raw_max_absdiff.max():.8g} | {z.feature_absdiff.max():.8g} |')
    lines+=['',
        '## 적용 범위와 한계','',
        '다른 날 전체 평균과 동일 시간 방법은 다른 날짜에도 생활 패턴이 반복된다는 가정이다. 중앙값은 이상값에 덜 민감하지만 사용량 합계나 드문 높은 조도를 보존하지 못할 수 있다. 평일/주말은 실제 근무일·공휴일을 알려주는 정보가 아니다. 최근 날짜 가중치의 반감기7일은 최적 기준이 아니다. 비슷한 날 복사는 남아 있는 같은 특징의 패턴만 보고 선택하며, 삭제 구간이 실제로 비슷하다는 보장은 없다.',
        '유효 donor 날짜는 해당 센서에 유효 원관측이 적어도 하나 있는 날짜이며, 하루 전체가 완전히 관측됐다는 뜻은 아니다. 시간대별 실제 donor 수와 donor 날짜별 자료량을 별도로 보고했다. 날짜에 동일 가중치를 주므로 관측 수가 적은 날짜도 해당 시간대에 관측이 있으면 한 날짜의 기여를 갖는다.',
        '전체 날짜 방법은 미래 날짜를 포함하는 오프라인 복원이다. 과거 방법도 당일 fallback에서는 이후 관측을 사용할 수 있다. 다른 평가 날짜는 현재 대상 날짜가 아닐 때 온전한 donor로 사용한다. 실제 여러 날짜가 동시에 손실되거나 자연 결측이 다른 원인으로 생기는 상황의 효과를 검증한 것은 아니다.',
        '모든 방법은 raw 조도에서 채운 다음 log1p 특징을 계산하고, 사용량은 raw ms를 채운 다음 분 단위 합계를 계산했다. 이진 값은0/1만 채웠다. 삭제된 원래 값·정답 레이블·reference·scale을 대치 또는 유사한 날 선택에 사용하지 않았다.',
        '여기서 평가한 것은 원래 관측에 기반한 일별 특징값의 왜곡이다. 특징 오차가 줄었다고 개별 삭제 관측의 실제 값까지 정확히 복구했다고 단정할 수 없다. 참가자10명의 개인 내 복원이며, 더 큰 집단의 예측 성능 검증을 대체하지 않는다. 신규 결과는 심사 후 탐색적 보완으로 표현해야 한다.','',
        '사용량은 비음수 관측의 합계이므로 누락만 적용하면 원래 관측 합계보다 작아진다. 대치의 이득에는 이 합계 부족을 줄이는 효과가 포함된다. 따라서 시간대 정보 자체의 추가 효과는 무대치 대비 이득만으로 판단하지 않고 다른 날 전체 평균 대조군과의 짝지은 비교로 평가한다.','',
        '## 파일 안내','',
        '- PROTOCOL_FINAL.md / FROZEN_PROTOCOL.json: 설정 근거·동결 기록.',
        '- donor_inventory/FEASIBILITY.md: 실제 donor 수·시간 정렬·원자료 대조.',
        '- full_01/recovery_rows.parquet: 신규40만행 원결과. audit_raw에는 고정 draw0의 실제 채움값.',
        '- summary_01/RESULTS.md: 기존·신규 모든 방법, 양수·음수·0·N/A를 포함한 상세 표.',
        '- summary_01/geometry_residual.csv: 모든 방법의 연속−무작위 잔여 오차.',
        '- figures_02/: 두 비교군 대비 효과와 참가자 bootstrap 구간을 그린 검토 완료 PNG·SVG. figures_01은 범례 위치 수정 전 초안을 보존한 것이다.',
        '- independent_audit_02/verification.json: 최종 독립 검사 범위·실측 차이·해시. independent_audit_01에는 첫 실패 기록을 보존했다.','']
    destination.write_text('\n'.join(lines),encoding='utf8')
    print(destination)
if __name__=='__main__':run()
