과거 준비 검토 기록입니다. 2026-10-05 작성자가 전체 연구 코드의 직접 작성을 확인했고, 이후 공개 배포 준비를 지시했습니다. 아래 출처 미확인 표현은 당시 상태입니다. 최신 실행은 verification/PUBLICATION_CHECK_20261005.md를 참고하세요.

# ICTC 2026 공개 준비: 남은 다섯 항목

작성일: 2026-10-04. 이번 작업은 파일·기존 실행 기록 확인과 문서 정리다. 테스트, 데이터 다운로드, 원격 업로드 및 장시간 연구 실행은 이번에 수행하지 않았다.

## 1. 포스터

사용자 제공 사실: A0(841×1189mm), 90×135cm 두 규격의 본문·레이아웃 및 인쇄용 PDF 검수는 별도 작업에서 완료됐다. 이 후보에는 실제 PNG/PDF가 없다. 코드 게시 대상이 확정되면 사용자가 실제 레포 링크 또는 QR을 넣은 최종 파일을 전달한다.

전달 후 파일명·규격·버전·내용·해시를 확인하고 docs/assets/에 연결한다. README 상단에는 폭 약 440px의 실제 PNG 미리보기와 각 규격의 PDF/PNG 링크를 추가한다. 현재는 미리보기·다운로드 링크를 제공하지 않는다. 외부 작업의 검수 완료와 이 후보에서의 파일 검증을 구분한다.

## 2. 레포 정보

- 사용자 제공 원격 주소: https://github.com/dansojo/ictc2026-missingness-stress-test.git
- 표시 이름: dansojo/ictc2026-missingness-stress-test
- 웹 주소: https://github.com/dansojo/ictc2026-missingness-stress-test
- 공개 범위: Public — 이전 사용자 발언 및 제공 화면에 근거. 이번에 원격 상태를 조회한 것은 아니다.
- 레포 생성은 사용자가 완료. 이 후보의 push·게시 완료나 공개 commit/tag는 아직 확인되지 않았다.

현재 안내의 미정 표시는 받은 정보로 갱신한다. 과거 검증 기록의 당시 미정/미게시 상태는 역사적 기록으로 보존한다. 원격 업로드는 별도 지시 후 진행한다.

## 3. 출처·라이선스 점검

후보 전체 180개 파일을 대상으로 라이선스 파일명과 SPDX, copyright, MIT/Apache/GNU 라이선스 문구, GitHub 원본 URL 표기를 검색했다. LICENSE/LICENCE/COPYING/NOTICE 파일 및 해당 명시적 라이선스·저작권·원본 GitHub URL 표기는 발견하지 못했다. 이는 외부 코드가 없다는 증명이 아니다.

SOURCE_MANIFEST.json의 136개 항목은 destination, bytes, sha256 및 일부 original_sha256만 제공한다. 출처 바이트의 보존 근거는 있지만 저작권자·원본 URL/commit·원본 라이선스·재배포 허가를 판정할 정보는 부족하다. 19개 수정 항목이 있다는 기록도 권리 허가를 뜻하지 않는다.

| 영역 | 현지 근거 | 남은 확인 |
|---|---|---|
| scripts/, tests/의 배포 보조 작업 | PROVENANCE.md가 후속 작성 작업으로 구분 | 실제 작성자·기여자 및 기관/공동연구 권리 확인 |
| original_paper_reproduction/와 extensions/ | 기존 연구 아카이브에서 선별한 136개 항목 및 해시 | 저자 직접 작성/외부 도입 여부를 파일별 확인 |
| paper/vendor, g2_vendor, full_vendor | 코드 포함, 현지 라이선스 고지 없음 | 폴더명만으로 외부 저작물이라고 단정하지 않음. 원본 출처와 권리 확인 우선 |
| requirements의 11개 직접 의존성 | 설치 요구사항이며 배포 후보에 설치본은 없음 | 실제 소스 복사/수정 여부 확인. 향후 의존성 번들 시 각 버전의 LICENSE/NOTICE 점검 |
| README·논문·포스터 | 연구 저자 표기, 최종 포스터 미입수 | 논문 출판 계약, 그림/로고/폰트 등 별도 권리와 재배포 조건 |
| ETRI·StudentLife·ExtraSensory | 원자료·참가자 기록 등 제외, 제공자별 접근 안내 | 제공자 이용 조건, 저자 아카이브/비공개 참조자료의 제공 권한 |

따라서 현재 결론은 **라이선스 선택 보류, 외부 코드 권리 확인 미완료**다. 다음 장부를 채운 뒤 선택한다: 파일/범위 → 직접 작성 또는 외부 도입 → 원저작자 → 원본 URL와 commit/version → LICENSE/NOTICE 원문 → 수정 내역 → 재배포 조건 → 확인 근거. 원본 주소를 알 수 없는 항목은 저자의 원래 아카이브·기여 이력·계약을 확인한다. 무고지 외부 코드는 임의 MIT 재허가하지 않고, 허가 확보 또는 제외/대체를 검토한다.

조건부 선택지:

- **MIT**: 소유권과 외부 코드 호환성을 확인한 자체 코드에 대해 단순한 재사용 허용을 원할 때. 저작권·라이선스 고지 보존이 필요하다. https://choosealicense.com/licenses/mit/
- **Apache-2.0**: 같은 전제 아래 명시적인 특허 허여와 변경 고지를 원할 때. 기존 NOTICE 등 보존 조건도 확인한다. https://choosealicense.com/licenses/apache-2.0/
- **GPL-3.0**: 파생 코드 배포 시 소스와 같은 라이선스 제공을 요구하려는 경우. 실제 외부 라이선스의 호환성과 결합 범위를 먼저 확인한다. https://choosealicense.com/licenses/gpl-3.0/

쉬운 연구 코드 재사용이 목적이고 권리 확인이 완료된다면 MIT를 우선 후보로 검토할 수 있지만, 지금 확정할 수는 없다. 코드 라이선스 범위에서 논문·포스터·데이터를 명시적으로 분리하고 각 권리는 별도 표기한다. Public이라는 사실만으로 일반 재사용 허가가 생기지 않는다. https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository

## 4. 기존 검증 기록

인용 기준은 2026-10-04 레이아웃 재검증이다. 공개 244개 발견 중 **238개 통과·6개 조건부 제외**, 별도 비공개 fixture **6개 통과**다. 이전 10월 2일/3일의 다른 테스트 개수와 혼합하지 않는다.

- 환경: CPython 3.12.14, Windows, 격리 인터프리터의 .pth로 기존 설치 라이브러리를 연결. 새 환경의 온라인 설치 성공 기록이 아니다. 구조화 기록의 Windows 11 빌드 값은 이전 fresh 환경의 기록이며, 10월 4일 실행에 별도로 기록되지 않은 상세 환경/시각을 추정하지 않는다.
- 직접 의존성 고정 목록: numpy 2.4.6, pandas 3.0.3, pyarrow 24.0.0, matplotlib 3.10.9, Pillow 12.2.0, pypdf 6.10.0, scipy 1.17.1, scikit-learn 1.9.0, lightgbm 4.6.0, joblib 1.5.3, threadpoolctl 3.6.0. 전체 전이 의존성 lockfile은 아니다.
- 실행 날짜는 2026-10-04 기록으로 확인. 정확한 시작 시각/시간대는 해당 요약에 없으므로 쓰지 않는다.
- 제외 이유: 공개 후보에 원본 adapter fixture, 비교 표·표시 fixture, canonical/primitive fixture가 없거나 설정되지 않음. ICTC_PRIVATE_REFERENCE, ICTC_TEST_ROOTS, FULL_DOWNSTREAM_ORIGINAL_TEST_ROOT가 관련된다.
- 별도 6개는 adapter/deletion guard, 8개 CSV 수식, 값 변경 시 prose/design, 고정 표시 수치, display 렌더링, small stress replay. 잘못된 비공개 source root로 최초 1개 import 실패 후 설정만 수정해 6개 통과. 수정된 실행 로그는 41.117초다.
- 기록: docs/verification/LAYOUT_RECHECK_20261004.md 및 verification.json의 layout_recheck_20261004. 비공개 public-tests.log 및 six-private-fixtures-corrected.log도 읽어 확인했다. 비공개 로그·경로는 공개 후보에 복사하지 않는다.

공개 재확인 명령은 `python -B scripts/run_tests.py`, `python -B scripts/verify_release.py`다. 이번에는 실행하지 않았다. 이번 변경은 문서만이다. 이후 코드 변경 시 영향 모듈 테스트·CLI 실행을 새 실행 기록으로 남기고, 배포 직전 공개 suite/audit를 재실행한다. 기존 기록은 덮어쓰지 않는다.

## 5. 전체 연구 재현: 해결 순서

현재는 테스트·제한 실행·과거 집계 비교가 확인됐을 뿐 전체 raw-to-paper 재현 완료가 아니다. 30개 논문 값은 과거 집계와 인쇄 정밀도 비교다. 10월 4일 same-day 1,000행, cross-day 800행의 제한 복원은 exact 일치, 외부 예측은 2 fit/74 window의 제한 실행이다. 전 연구 결과로 확대하지 않는다.

| 순서 | 필요한 입력과 작업 | 시간 근거/제안 |
|---|---|---|
| 1 | 새 CPython 3.12 환경에 고정 requirements 설치, 기존 .pth 재사용 없이 import 확인, 전체 설치 목록 기록. Linux 검증은 별도 | 계획상 15–45분 예산, 실측 아님. 기존 PyPI 접속은 WinError 10013 실패; 네트워크 허용 조건 확인 필요 |
| 2 | 원래 허가된 저자 archive, frozen protocol, private reference 및 raw data 확보. source_roots를 현재 후보에 연결; 데이터/출력은 repo 밖에 배치 | 자료 확보 시간 미정. 보유분 gate 검사는 기존 수초 단위, 전체 해시는 저장장치와 크기에 따라 달라짐 |
| 3 | ExtraSensory 원본 cv5Folds.zip(10,803 bytes, SHA256 66183635452d8c1b250b570fba3000296bf29e4d54eda6af270d2b4f5a3b9ee6)과 provider README.txt(6,692 bytes, SHA256 a5d222248abd0a2380b832be2bbec316812c7ca4b9c7831a6ae065b481ffa8c0) 복구·검증 | 확보 시간 미정. exact 원본 endpoint는 미확인. 기존 원본 사본 우선; 새 다운로드는 별도 지시 필요 |
| 4 | 공개 suite/audit, 각 input gate, 제한 smoke와 9단계 fresh chain으로 환경·입력·실제 stage 부모 연결 확인 | 이전 공개 suite 약 13초(10월 3일), private fixture 약 41초(10월 4일), smoke 각 약 5–7초(10월 4일), 5-cell fresh chain 약 505초(10월 2일). 동일 시간 보장 아님 |
| 5 | 별도 지시 후 새 비공개 출력 루트에서 base 전체 및 M0/legacy/same-day, 50-partition donor rebuild, cross-day, external inventory와 102 external fits, summaries/bootstrap/audit 실행 | 역사적 base 약 11시간, 추가 기록 단계 합계 41.9분. inventory·summary·bootstrap·검사·오버헤드 제외라 전체 총시간으로 사용하지 않음. 최소 반나절 이상 창을 확보하고 미측정 단계를 더함 |
| 6 | 생성한 모든 논문 표/CI/그림을 최종 원고와 대조, 실패와 exact/tolerant 진단 구분 | 계획상 1–3시간 수동 검토 예산, 계산 시간 별도·미실측. Figure 1은 최종 원고의 inline TikZ이며 구형 fig2 렌더링과 구분 |

엄격 Brier 비교는 과거 false/exit 2(차이 최대 8.326672684688674e-17). preparation 전체 열 exact 비교는 false(최대 1.7763568394002505e-15). 수치가 작다는 이유로 성공으로 바꾸거나 tolerance·freeze·seed를 바꾸지 않는다. full downstream fitting, donor 전체 재생성, external inventory 재생성, 102 external fits 및 전체 논문 요약의 새 실행은 미완료다. 제공자 데이터만으로 현재 launcher 전체를 재현할 수 있다고 주장하지 않는다.

이 계획은 실행 승인이 아니다. 장시간 실행과 원격 업로드는 별도 지시까지 대기한다.
