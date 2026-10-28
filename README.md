# VC-SASR -- Variable-Channel Spatial ASR Benchmark

**1~4개 동기 마이크를 하나의 가중치로 처리해, 전사와 세션 내 익명 화자 ID, 방위각·거리의
확률 분포와 점 추정을 함께 출력하는 모델을 위한 벤치마크.** 채널 수가 런타임에 바뀌어도
모델을 바꾸지 않는다는 것이 이 과제의 전제다.

왜 분포를 요구하는가: 어레이 반경 상한이 0.12m, 즉 최대 쌍 간격 0.24m라서 공간
에일리어싱 한계가 `spec.ALIASING_LIMIT_HZ` = **714.6Hz**다. 음성 에너지 대부분이 그 위에
있으므로 좁은 어레이는 협대역 위상차만으로 방위를 유일하게 풀 수 없다. 특히 M=2에서는
방위 사후분포가 본질적으로 **양봉(bimodal)** 이다. 점 추정만 받으면 이 구조가 측정되지
않으므로, 방위 36빈·거리 8빈 분포를 제출받아 NLL과 HPD 커버리지로 채점한다.

---

## 현재 상태 (2026-10-29)

**정직하게**: 이 저장소는 기획안 v1.7 기준으로 **평가 기반만 올라간 상태**다.

아래 "통과"·"막혀 있음" 은 모두 2026-10-29 에 **실제로 실행해서** 확인한 결과다.

| 항목 | 상태 |
|---|---|
| 평가 계약 `spec.py` (`SPEC_VERSION 0.2.0`) | 올라감. doctest 통과, 자체 모순 검사 통과 |
| 공식 평가기 `evaluate.py` | 올라감. CLI 는 `--pred / --gt / --track / --out` |
| 세션 설계·합성 라이브러리 `simulate.py` | 올라감. 라이브러리 + 자체점검 |
| eval-dev 생성기 `make_eval_dev.py` | 올라감. 데이터셋을 쓰는 엔트리포인트는 이쪽이다 |
| 베이스라인 1·2 (학습 불필요, 추론 전용) | 올라감. `baselines.single_channel`, `baselines.dsp_srp_phat` |
| `make test` | **통과.** doctest 5개 파일 + `tests/` 유닛테스트 100개 |
| `make selftest` | **통과.** simulate 불변식 + 베이스라인 1·2 종단 자체점검 |
| `make eval-dev` | **통과.** 오프라인 합성 경로로 4세션(24케이스) 생성 확인 |
| `make baseline` | **막혀 있음.** `meta.jsonl` 의 `audio` 스키마 불일치 |
| `make score` | **통과.** 생성된 정답 24케이스를 실제로 채점해 `scores.json` 생성 확인 |
| **Recipe 1 모델 학습** | **미실시** |
| **리더보드 전 지표** | **미측정.** 아래 표의 모든 칸이 비어 있다 |
| Track 2 (이동·다중 화자) | 범위 외. 12/7 이후 증분 |
| Unseen Room 일반화 | 범위 외 |

즉 **지금 이 저장소에 점수는 하나도 없다.** README나 발표자료에 숫자를 적으면
그건 측정값이 아니다.

점수가 처음 생기려면 `make eval-dev` -> `make baseline` -> `make score` 가 한 번
돌아야 한다. **생성과 채점은 돌고, 중간의 추론 한 칸이 막혀 있다.** 빈 제출
파일로 `make score` 를 돌리면 24케이스 전부에 R-MISS 페널티가 붙은 `scores.json`
이 나오므로(`azimuth_mae_deg = 180.0`, `cpwer = N/A by rule`), 채점 경로 자체는
정답 데이터에 대해 확인된 상태다.

남은 블로커 하나:

- `make baseline` -- 생성기는 `meta.jsonl` 의 `audio` 를 **객체**로 쓰는데
  (`{"path": "audio/sess_0000.wav", "sample_rate_hz": 16000, ...}`) 베이스라인
  로더는 그 값을 **경로 문자열**로 읽는다. 그래서
  `data/eval-dev/{'channels_in_file': 4, ...}` 를 열려다 `FileNotFoundError` 가
  난다. `spec.py` 에 per-case 매니페스트 스키마가 아직 없어서 생긴 불일치이고
  (제출·정답 레코드 스키마는 v0.2.0에서 동결됐다),
  CLI 로는 우회할 수 없다 -- 두 모듈 중 한쪽을 맞춰야 한다.

진단과 고쳐야 할 쪽은 `SPEC_DECISIONS.md` 14절 2항에 적어 두었다.

학습이 필요한 모델은 이번 회차에 **없다.** 베이스라인 1·2는 둘 다 추론 전용이므로
GPU 없이, 체크포인트 다운로드 없이도 파이프라인 전체가 한 번 돈다(베이스라인 1의 동결
ASR 백엔드만 최초 1회 가중치가 필요하다).

---

## 일정 -- 모델 학습 기한 **2026-12-07**

오늘 기준 기한까지 **39일**. 주차 계획과, 각 주가 밀렸을 때 **무엇을 버리는지**를 같이
적는다. 버릴 것을 미리 정해두지 않으면 마지막 주에 전부 반쯤 된 상태로 남는다.

| 기간 | 일수 | 산출물 | 늦어지면 무엇을 버리는가 |
|---|---|---|---|
| **10/29 (목) ~ 11/03 (화)** | 6 | 평가 계약 동결, `evaluate.py`, `simulate.py`, eval-dev 생성, 베이스라인 1·2 수치 확보 | eval-dev 세션 수를 150 -> 40으로 줄인다. **마스크 6개는 줄이지 않는다** -- 마스크를 깎으면 채널 수 효과라는 벤치마크의 본론이 사라지므로, 세션 수를 먼저 깎는다 |
| **11/04 (수) ~ 11/17 (화)** | 14 | Recipe 1 학습 (가변 채널 단일 가중치), causal 추론 경로 완성, 제출 파일 생성 | Recipe 2·3를 전부 버리고 Recipe 1 단일 모델만 제출한다. 레시피 비교 표는 "미실시"로 남긴다. MVS는 그래도 성립한다 |
| **11/18 (수) ~ 11/30 (월)** | 13 | Ablation: Exp 1(채널 수 M=1..4 효과), Exp 2(청크 간 상태 유지 vs 초기화) | Exp 2를 버린다. Exp 1만 남긴다. Exp 1은 이 벤치마크의 핵심 주장이고 Exp 2는 구현 선택에 대한 부가 실험이다 |
| **12/01 (화) ~ 12/07 (월)** | 7 | 리더보드 확정, HTML 결과 페이지, 저장소·PR 정리, 피어 검토 반영 | Unseen Room 일반화 표와 `windowed`/`full_audio` 비공식 표를 버리고, `causal` x 마스크 6개 공식 표 하나만 낸다 |

**오늘(10/29) 안에 끝내야 하는 것**: 저장소 구성, 데이터 준비 경로, 베이스라인 업로드.
학습은 오늘의 범위가 아니다.

---

## MVS -- 최소 성립 제출물

이 조합만으로 **HTML 결과 페이지, 리더보드, 저장소, PR이 전부 성립한다.**

```
Track 1 (단일 정지 화자)
  x  observation_mode = causal          (spec.OFFICIAL_SCOPES)
  x  spec.EVAL_MASKS 6개                (1000 / 0010 / 1100 / 1010 / 1110 / 1111)
  x  Baseline 1, Baseline 2             (학습 불필요)
  =  공식 표 1개 + spec.OFFICIAL_METRICS 15개 중 Track 1 해당 14개
     (cpwer 는 spec.TRACK_UNDEFINED_METRICS 에 따라 규칙상 N/A)
```

MVS가 성립하는 근거는 이렇다. 케이스 키는 `session_id x mask` 6개이고, 세션 길이 20초 /
그리드 0.25초이므로 케이스당 채점 시점이 80개다. 즉 세션 하나가 **공간 레코드 480개**를
만들고, 이것만으로 NLL·커버리지·MAE·정확도가 모두 계산된다. 학습 모델이 하나도 없어도
표는 채워진다.

**위에 덧붙이는 증분** (MVS가 아니다, 하나도 없어도 제출은 성립한다):

- Track 2 (이동 화자 / 2화자) -- 여기서 처음 `cpwer`가 의미를 가진다
- Unseen Room 일반화
- Recipe 2 / Recipe 3
- `windowed` / `full_audio` 관측 범위 -- 값은 받지만 `W`가 고정되지 않아 비공식 표로만 간다

이 우선순위를 뒤집지 마라. 증분을 먼저 하면 밀렸을 때 버릴 것이 MVS밖에 남지 않는다.

---

## 5분 재현

```bash
git clone <this-repo> vcsasr
cd vcsasr

# 1) 점수만 재계산할 거면 이것만으로 충분하다 (numpy 하나)
python3 -m pip install numpy==2.2.4

# 2) 데이터 생성과 베이스라인까지 돌릴 거면 전체 설치
python3 -m pip install -r requirements.txt

# 3) 계약과 문법 검증 (무거운 의존성 없이 통과해야 한다)
make test              # 통과 확인됨
make selftest          # 종단 불변식까지. 통과 확인됨 (합성 오디오, 네트워크 불필요)

# 4) eval-dev 합성 -> 베이스라인 추론 -> 채점
make eval-dev          # data/eval-dev/ 에 audio/ + ground_truth.jsonl + meta.jsonl
make baseline          # predictions.jsonl   <- 지금 audio 스키마 불일치로 막혀 있다
make score             # scores.json
```

공식 `make eval-dev` 는 LibriSpeech 와 `pyroomacoustics` 가 필요하다. 둘이 없는
환경에서도 파이프라인을 한 바퀴 돌려보려면 합성 경로를 쓴다 (네트워크 불필요).
이 산출물에는 `official=false` 가 찍히므로 **보고에 쓰지 마라.**

```bash
make eval-dev N_SESSIONS=4 EVAL_DEV_ARGS="--synthetic --rir-engine fallback_ism"
make baseline BASELINE_ARGS="--synthetic-asr"
make eval-dev-verify N_SESSIONS=4   # 렌더링 없이 불변식만 (numpy 불필요)
```

지금 확실히 통과하는 것은 아래 다섯 개다 (전부 실행해서 확인).

```bash
make test              # doctest 5개 파일 + tests/ 유닛테스트 100개
make selftest          # 베이스라인 1·2 를 합성 세션에 대해 종단 실행
make sim-selftest      # simulate.py 의 plan-layer 불변식만
make eval-dev N_SESSIONS=4 EVAL_DEV_ARGS="--synthetic --rir-engine fallback_ism"
make score             # 빈 제출 파일이어도 R-MISS 페널티로 scores.json 이 나온다
```

`make` 변수는 전부 커맨드라인에서 덮어쓸 수 있다 (`TRACK`, `SCOPE`, `BASELINE`,
`N_SESSIONS`, `SEED`, `DATA_DIR`, `GT`, `PRED`, `SCORES`, `SIM_SESSIONS`).
각 타깃의 설명은 `make help`.

> **CLI 정합 상태** (2026-10-29, `--help` 로 실제 확인):
>
> - `make baseline` -- **정합.** 베이스라인 1·2 모두 `--eval-dir / --out / --scope /
>   --track` 을 받는다. `BASELINE=2` 의 모듈명은 `baselines.dsp_srp_phat` 이다
>   (`baselines.srp_phat` 이 아니다).
> - `make score` -- **정합.** `evaluate.py` 의 플래그는 `--pred / --gt / --track /
>   --out / --indent` 다. 정답 파일 플래그는 `--labels` 가 아니라 **`--gt`** 이고,
>   **`--scope` 플래그는 없다**: 평가기가 각 레코드의 `observation_mode` 를 읽고
>   `spec.OFFICIAL_SCOPES` 로 직접 공식/비공식 표를 나눈다. 그래서 `SCOPE` 는
>   `make baseline` 에만 쓰인다.
> - `make eval-dev` -- **정합.** 데이터셋을 쓰는 엔트리포인트는 `simulate.py` 가
>   아니라 **`make_eval_dev.py`** 다 (`--out / --n-sessions / --seed / --synthetic /
>   --rir-engine / --verify-only`). `--track` 플래그는 없다: eval-dev 는 Track 1
>   구성이고 트랙 선택은 추론·채점 쪽에서만 의미가 있다. `simulate.py` 는
>   세션 설계·합성 라이브러리 + 자체점검이고 `--out` 이 없다.
> - `MANIFEST` -- 베이스라인에 `--manifest $(DATA_DIR)/meta.jsonl` 을 **명시해서**
>   넘긴다. 자동탐색 후보에 `meta.jsonl` 이 없어서 데이터셋 요약 파일인
>   `manifest.json` 을 먼저 집어 터지기 때문이다 (13절 9항 (a)).
>
> `SCOPE` 의 기본값은 리터럴이 아니라 `spec.OFFICIAL_SCOPES[0]` 에서 끌어온다
> (하드코딩 금지). `GT` 의 기본 파일명 `ground_truth.jsonl` 은 생성기가 실제로
> 쓰는 파일명이고 `spec.py` 에는 없다 -- `SPEC_DECISIONS.md` 14절 3항.

---

## 제출 스키마

제출 파일은 **JSONL**이다 (레코드 하나가 한 줄). 레코드는 두 종류다. 아래 두 예시는
`spec.py`로 실제 검증한 값이다 -- `azimuth_probs` 길이 36, `distance_probs` 길이 8,
두 벡터 모두 합이 정확히 1.0이다.

### 레코드 1: `spatial_frame` -- 그리드 시점 하나의 공간·화자 예측

케이스당 80개, 세션당 480개 (마스크 6개). 가독성을 위해 아래는 들여쓰기해 두었지만
실제 파일에서는 **한 줄**이다.

```json
{
  "record_type": "spatial_frame",
  "case_id": "sess_0001__m1010",
  "observation_mode": "causal",
  "eval_time_sec": 7.25,
  "speaker_id": "s0",
  "speech_active": true,
  "azimuth_probs": [0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.00031, 0.000757, 0.006822, 0.043523, 0.134681, 0.196431, 0.134681, 0.043534, 0.00717, 0.005653, 0.032731, 0.101084, 0.147394, 0.101084, 0.032716, 0.00519, 0.000641, 0.000306, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294],
  "distance_probs": [0.023962, 0.095004, 0.226493, 0.303436, 0.226493, 0.095004, 0.023962, 0.005646],
  "azimuth_point_deg": -36.1,
  "distance_point_m": 1.79
}
```

이 예시가 왜 이렇게 생겼는지: 마스크 `1010`은 채널 0과 2를 쓰는 **최장 쌍 M=2**다. 좌우
모호성 때문에 방위 분포가 빈 14 (중심 -35도)와 빈 21 (중심 +35도)에 두 봉우리로 갈린다.
`spec.INTERVAL_RULE`의 HPD@90 집합은 `{12,13,14,15,16, 20,21,22}`로 **연속이 아니고**,
폭 80도에 누적질량 0.9024다. 참 방위 -37.4도는 빈 14이므로 커버리지 1, NLL 1.6274다.
이게 이 벤치마크가 측정하려는 바로 그 구조다.

### 레코드 2: `final_transcripts` -- 케이스 하나의 최종 전사

케이스당 정확히 1개, 세션당 6개.

```json
{
  "record_type": "final_transcripts",
  "case_id": "sess_0001__m1010",
  "observation_mode": "causal",
  "final_transcripts": {"s0": "the quick brown fox jumps over the lazy dog"},
  "rtf": 0.42
}
```

### 필드 사양

`spatial_frame`:

| 필드명 | 타입 | 필수 | 단위 | 채점 대상 지표 | 누락 시 동작 |
|---|---|---|---|---|---|
| `record_type` | string = `"spatial_frame"` | 권장 | - | - | `azimuth_probs` 유무로 추론. 모호하면 거부 |
| `case_id` | string `<session_id>__m<mask_bits>` (`spec.make_case_id`) | 필수 | - | 모든 지표의 1차 키 | 레코드 거부 (스키마 오류) |
| `observation_mode` | string, `spec.OBSERVATION_MODES` 중 하나 | 필수 | - | 보고 범위 분리 (`causal`만 공식) | 레코드 거부. 기본값으로 `causal`을 가정하지 않는다 |
| `eval_time_sec` | float, `k * spec.DELTA_T_SEC` (0.25의 정수배) | 필수 | 초, 원점은 세션 첫 샘플 (`spec.TIME_ORIGIN`) | 모든 공간·화자 지표의 2차 키 | 레코드 거부 |
| `speaker_id` | string (세션 내 익명 슬롯, Track 1은 `spec.MAX_SPEAKERS_PER_TRACK[1]` = 1개) | 필수 | - | `speaker_attribution_accuracy` | `spec.PENALTY_SPEAKER_CORRECT` = `False` (오답 처리) |
| `speech_active` | bool | 선택 | - | 채점하지 않음 (`T_eval`은 정답 기준으로 고른다) | 무시 |
| `azimuth_probs` | array[`spec.K_AZIMUTH`=36] of float | 필수 | 확률, 합 1 (`spec.PROB_SUM_TOLERANCE`=1e-4) | `azimuth_nll`, `azimuth_coverage_90`, `azimuth_interval_width_deg` | `spec.PENALTY_NLL` ≈ 27.631 + `spec.PENALTY_AZIMUTH_DEG` = 180.0 |
| `distance_probs` | array[`spec.K_DISTANCE`=8] of float | 필수 | 확률, 합 1 | `distance_nll`, `distance_coverage_90`, `distance_interval_width_m` | `spec.PENALTY_NLL` + `spec.PENALTY_DISTANCE_M` = 5.0 |
| `azimuth_point_deg` | float, `[-180, 180)` | 선택 | 도 | `azimuth_mae_deg`, `azimuth_acc_10deg`, `azimuth_acc_20deg`, `pos_error_2d_m` | MAP 빈의 `spec.AZIMUTH_BIN_CENTERS_DEG` 값을 대표점으로 사용 (R-PT). 거부하지 않는다 |
| `distance_point_m` | float, `[0.5, 5.0]` | 선택 | m | `distance_mae_m`, `pos_error_2d_m` | MAP 빈의 `spec.DISTANCE_BIN_CENTERS_M` (기하 중심) 사용 (R-PT) |

`final_transcripts`:

| 필드명 | 타입 | 필수 | 단위 | 채점 대상 지표 | 누락 시 동작 |
|---|---|---|---|---|---|
| `record_type` | string = `"final_transcripts"` | 권장 | - | - | `final_transcripts` 키 유무로 추론 |
| `case_id` | string, 위와 동일 | 필수 | - | `wer`, `cpwer`의 키 | 레코드 거부 |
| `observation_mode` | string, 위와 동일 | 필수 | - | 보고 범위 분리 | 레코드 거부 |
| `final_transcripts` | object `{speaker_id: string}`, 키 수 <= `spec.MAX_SPEAKERS_PER_TRACK[track]` | 필수 | - | `wer` (`spec.WER_AGGREGATION` = micro_global), `cpwer` | `spec.MISSING_TRANSCRIPT_POLICY` = `count_all_deletions`: 참조 단어 전부를 삭제 오류로 센다. 케이스를 드롭하지 않는다 |
| `rtf` | float (무차원, 연산시간/오디오시간) | 권장 | 배수 | `rtf` | 리더보드의 `rtf`를 N/A로 남긴다 (0으로 채우지 않는다) |

**케이스당 레코드 수**: `spatial_frame` 80개 + `final_transcripts` 1개.
세션당 `(80 + 1) x 6 = 486`개.

**필드명의 출처**: 전부 `spec.py`다 (`F_*` 상수, `FIELD_ALIASES`, `REQUIRED_FIELDS`,
`PROB_FIELD_LENGTHS`). v0.1.0까지는 이 README가 정의한 관례였고 세 모듈이 각자
갈라져 있었다. 정식 `record_type`은 `timepoint` / `final_transcripts`이고, 예전
철자 `timestep` / `final_transcript`는 `spec.RECORD_TYPE_ALIASES`에 남아 **읽기만**
된다. 수용되는 철자와 그 해석 순서는 `SPEC_DECISIONS.md` 13절 참조.

---

## 채점 규칙 요약

각 규칙에는 ID가 있다. **`evaluate.py`가 던지는 예외 메시지와 이 문서가 같은 ID를 쓴다**는
것이 규약이다 (예: `R-NLL: azimuth_probs sums to 0.912, tolerance 1e-04`). 제출이 거부되면
메시지의 ID로 이 표를 찾아보면 된다.

| ID | 규칙 | 근거 상수 |
|---|---|---|
| **R-DUP** | `(case_id, eval_time_sec, observation_mode)`는 제출 파일 안에서 유일해야 한다. 중복이면 **거부**하며 "마지막 레코드 승리"로 처리하지 않는다. 암묵 덮어쓰기는 어느 레코드가 채점됐는지 알 수 없게 만든다 | `spec.DELTA_T_SEC` (0.25는 이진수로 정확히 표현되므로 `eval_time_sec`을 딕셔너리 키로 안전하게 쓸 수 있다) |
| **R-MISS** | 결측은 **드롭이 아니라 페널티**다. 방위 180.0도, 거리 5.0m, 화자 오답, NLL은 클리핑 바닥값 약 27.631. 전사 결측은 참조 단어 전량 삭제. 어떤 제출도 결측보다 나쁠 수 없으므로 "모르면 비워두기"가 이득이 되지 않는다 | `spec.PENALTY_AZIMUTH_DEG`, `PENALTY_DISTANCE_M`, `PENALTY_SPEAKER_CORRECT`, `PENALTY_NLL`, `MISSING_TRANSCRIPT_POLICY` |
| **R-PERM** | 예측 화자와 참조 화자를 `max(len(pred), len(ref))`까지 빈 전사로 패딩한 뒤 1:1 최적 배정을 찾는다. 짝 없는 예측 화자의 단어는 삽입, 짝 없는 참조 화자의 단어는 삭제. 동률은 min_cpwer -> max_speaker_match_count -> lexicographic 순서로 깬다. Track 1에서 `cpwer`는 규칙상 N/A | `spec.SPEAKER_PAD_TOKEN`, `PERMUTATION_TIE_BREAK`, `MAX_SPEAKERS_PER_TRACK`, `TRACK_UNDEFINED_METRICS` |
| **R-NLL** | 분포 길이는 정확히 36 / 8이어야 하고 합이 1e-4 안에 들어야 한다. 평가기는 **재정규화하지 않는다** -- 틀리면 `ValueError`로 거부한다. NLL 계산 시 바닥값 1e-12는 **참 빈에만** 적용되므로 제출 벡터 자체는 변형되지 않는다 | `spec.K_AZIMUTH`, `K_DISTANCE`, `PROB_SUM_TOLERANCE`, `NORMALISE_SUBMITTED_PROBS` (=False), `NLL_EPSILON` |
| **R-PT** | 점 추정은 선택이다. 없으면 MAP 빈의 대표점(방위는 빈 중심, 거리는 기하 중심)을 써서 MAE와 정확도를 계산한다. 즉 분포만 내도 모든 점 지표가 채워진다 -- 생략은 거부 사유가 아니다 | `spec.AZIMUTH_BIN_CENTERS_DEG`, `DISTANCE_BIN_CENTERS_M` |
| **R-TXT** | 가설과 참조 **양쪽**에 같은 정규화를 같은 순서로 적용한다: NFKC -> casefold -> 구두점 제거 -> 숫자를 단어로 -> 공백 정리. WER은 전역 마이크로 평균(총 편집거리 / 총 참조 단어)이고, 공간 지표는 케이스 macro 평균이다 -- **두 집계 방식이 다르다** | `spec.TEXT_NORMALISATION`, `WER_AGGREGATION`, `SPATIAL_AGGREGATION` |

> 규칙 ID는 `spec.REJECTION_RULES`(거부 7종)와 `spec.SCORING_RULES`(채점 6종)에
> 있다. 평가기의 `RULE_*` 상수는 거기 없는 ID를 쓰면 import 시점에 죽는다.

집계 요약:

- **공간·화자 지표**: 케이스 안에서 시점마다 `1/|T_eval|`, 그다음 케이스마다 동일 가중
  (`spec.SPATIAL_AGGREGATION` = `macro_per_case`). 말이 많은 세션이 결과를 지배하지 않는다.
- **WER / cpWER**: 전역 마이크로 평균 (`spec.WER_AGGREGATION` = `micro_global`).
- **커버리지·구간폭**: HPD@90 그리디 (`spec.INTERVAL_RULE`). 집합이 연속일 필요가 없고,
  그게 M=2 양봉을 측정하는 핵심이다.

---

## eval-dev 배포 규격

### 용량 계산 (직접 계산한 수치)

4채널 / 16kHz / int16 PCM:

```
128,000 B/s  =  4ch x 16,000 Hz x 2 B    (= 128 kB/s, 125 KiB/s)
```

세션 길이는 `spec.SESSION_LENGTH_SEC` = 20.0초로 고정되므로:

| 항목 | 계산 | 용량 |
|---|---|---|
| 세션 1개 오디오 (4ch WAV) | `128,000 x 20` | **2.56 MB** (2.441 MiB) |
| 세션 1개 라벨 (80행 x 약 163 B) | 그리드 80점 | **13.0 kB** |
| 세션 100개 | | 256.0 MB |
| **세션 150개 (= 기본 `N_SESSIONS`)** | 오디오 384.0 MB + 라벨 2.0 MB | **386.0 MB** |
| 세션 195개 | | 499.2 MB (상한 직전) |

**오디오는 마스크별로 저장하지 않는다.** 4채널 세션을 한 번 저장하고 로더가
`spec.mask_to_channels`로 채널을 골라낸다. 마스크 6개의 채널 수 합은 13이므로, 마스크별로
따로 저장하면 **3.25배** 용량이 된다. 절대 그렇게 하지 마라.

**라벨 JSONL**도 Track 1에서는 마스크와 무관하다 (같은 세션, 같은 기하, 같은 정답). 디스크에는
세션당 80행만 두고 로더가 마스크 6개로 펼친다. 마스크별로 펼쳐 저장하면
세션당 480행 x 163 B = 78.2 kB, 150세션에 **11.74 MB**가 되어 6배 낭비다.

| 라벨 저장 방식 | 세션당 | 150세션 |
|---|---|---|
| 세션당 80행 (**채택**) | 13.0 kB | 1.96 MB |
| 마스크별 480행으로 펼침 | 78.2 kB | 11.74 MB |

`predictions.jsonl`은 레코드가 629 B 정도(36+8 확률값 때문)라서 세션당 303 kB,
150세션에 **45.45 MB**다. git에 올리지 않는다 (`.gitignore`).

### 선언하는 상한

| 상한 | 값 | 근거 |
|---|---|---|
| **eval-dev 전체** | **<= 500 MB** | 세션 195개에 해당. GitHub 저장소 권고 1GB의 절반 안쪽에 두어 히스토리 증가분을 남긴다 |
| **단일 파일** | **<= 90 MB** | GitHub는 파일당 100MB를 **차단**하고 50MB에서 경고한다. 90MB를 자체 상한으로 둔다 |
| **tar 샤드 1개** | **<= 32 세션 (81.92 MB)** | 100MB 한계에는 39세션까지 들어가지만, 헤더와 여유를 두고 32로 끊는다 |
| 기본 `N_SESSIONS` | 150 (386.0 MB) | 상한까지 114 MB 여유 |

### 실제로 배포하는 것

챌린지 공지가 **"Evaluation용 데이터셋 필수 포함"**을 개인 GitHub 링크 안에 요구한다.
시드와 생성기는 데이터셋이 아니므로 오디오를 저장소에 같이 커밋한다.

| 경로 | 내용 | 커밋 |
|---|---|---|
| `data/eval-dev/audio/` | 144세션 4채널 WAV, 351.6 MB (개당 2.44 MB) | 예 |
| `data/eval-dev/ground_truth.jsonl` | 정답 라벨 5.43 MB | 예 |
| `data/eval-dev/meta.jsonl` | per-case 매니페스트 0.58 MB | 예 |
| `data/eval-dev/manifest.json` | 파일 목록 + SHA256 + 상수 0.05 MB | 예 |
| `data/eval-dev/rir/` | 15.0 MB. 채점이 읽지 않는 생성 내부물이고 같은 시드로 재생성된다 | 아니오 |
| `runs/*.jsonl` | 예측 파일. 베이스라인을 다시 돌리면 나온다 | 아니오 |

커밋 용량은 약 358 MB다. GitHub의 파일당 차단 100MB(개별 WAV 2.44MB)와 저장소 권고
1GB 모두 안쪽이다.

### Git LFS 를 쓰지 않는 이유

351.6 MB 바이너리는 LFS 의 교과서적 용례처럼 보이지만, **이 저장소에서는 반대로 작동한다.**

GitHub 무료 계정의 LFS 는 저장 1GB, **대역폭 월 1GB**다. 저장은 들어가지만 대역폭은
clone 할 때마다 소모되고 그 비용은 저장소 주인에게 붙는다. 351.6 MB 데이터셋이면 한 달에
세 번째 clone 에서 쿼터가 끊기고, GitHub 는 그 시점부터 LFS 다운로드를 **차단한다.**

채점자가 받아가야 하는 데이터셋이 "받을 수 없음"이 되는 쪽이, 저장소가 358 MB 인 것보다
나쁘다. 평범한 git 은 느릴지언정 끊기지 않는다.

비용은 하나다. 재생성할 때마다 히스토리에 352 MB 가 영구히 쌓인다. 세트가
`official: true` 이고 시드가 `20261029` 로 고정이라 다시 만들 일이 거의 없다는 전제를
깔고 받아들인 비용이다. 이 전제가 깨지면 (spec 이 바뀌어 라벨이 무효가 되는 등)
그때는 히스토리를 다시 쓰거나 Release 자산으로 옮긴다.

`data/**/rir/` 와 `data/eval-test/` 는 `.gitignore` 에 남아 있다. 전자는 내부물이고
후자는 라벨을 숨기는 것이 존재 이유라서 커밋하면 의미가 없어진다.

---

## 라이선스 · 출처 표기

### LibriSpeech 파생물

소스 음성은 LibriSpeech이고 **CC BY 4.0**이다. 즉 우리가 만든 합성 다채널 오디오는
**파생물이며, 재배포할 때 출처 표기 의무가 따라온다.** 실내 전파를 시뮬레이션해 섞어도
의무는 사라지지 않는다. eval-dev / eval-test를 어디에 올리든 다음을 같이 배포한다.

```
This dataset contains audio derived from LibriSpeech (Panayotov et al., 2015),
distributed under CC BY 4.0. The derived multichannel audio was produced by
simulated room acoustics and is redistributed under the same license.
Source: https://www.openslr.org/12
```

### 체크포인트 · 라이브러리 라이선스

**이번 회차는 네트워크를 쓰지 않았으므로 아래 라이선스를 하나도 재확인하지 않았다.**
전부 `[미확인]`으로 표시한다. 적힌 값은 각 프로젝트에서 통상 알려진 라이선스이고,
**배포 전에 해당 모델 카드와 저장소 LICENSE 파일에서 직접 확인하고 이 표를 갱신해야
한다.** 특히 gated 체크포인트는 조건 동의가 필요할 수 있다.

| 자원 | 이번 회차 사용 | 통상 알려진 라이선스 | 확인 상태 |
|---|---|---|---|
| LibriSpeech 코퍼스 | 소스 음성 | CC BY 4.0 (출처 표기 필수) | **[미확인]** -- 기획안 기준, 재확인 필요 |
| `pyroomacoustics` | RIR 생성 | MIT | **[미확인]** |
| OpenAI Whisper (코드·가중치) | 동결 ASR 후보 | MIT | **[미확인]** |
| `facebook/wav2vec2-*` | 동결 ASR 후보 | Apache-2.0 | **[미확인]** |
| `transformers` | 백엔드 로더 | Apache-2.0 | **[미확인]** |
| `torch` / `torchaudio` | 추론 | BSD-3-Clause | **[미확인]** |
| `numpy` / `scipy` | 평가·DSP | BSD-3-Clause | **[미확인]** |
| `soundfile` | WAV 입출력 | BSD-3-Clause (단, 의존하는 libsndfile은 LGPL 계열) | **[미확인]** -- 정적 링크 배포 시 특히 확인 필요 |
| SpeechBrain / ECAPA-TDNN | **미사용** (의존성 충돌로 제외) | Apache-2.0 | **[미확인]** |
| `pyannote.audio` 세그멘테이션 | **미사용** | gated, 조건 동의 필요 | **[미확인]** |
| NVIDIA NeMo | **미사용** (의존성 충돌로 제외) | Apache-2.0 | **[미확인]** |

확인 담당과 시점: 리더보드 확정 주(12/01~12/07)에 실제로 사용한 체크포인트만 확인하고
`[미확인]`을 지운다. **사용하지 않은 자원은 확인하지 않고 "미사용"으로 남긴다.**

---

## 저장소 구성

```
spec.py                      동결된 평가 계약. 모든 상수의 유일한 출처. 수정 시 SPEC_VERSION 필수
SPEC_DECISIONS.md            기획안 v1.7에 없던 값을 spec.py가 어떻게 닫았는가 (피어 검토용)
evaluate.py                  공식 평가기 (--pred / --gt / --track / --out)
simulate.py                  세션 설계·합성 라이브러리 + 자체점검 (--self-test). 데이터셋은 쓰지 않는다
make_eval_dev.py             eval-dev 생성 엔트리포인트 (--out / --n-sessions / --seed / --verify-only)
baselines/single_channel.py  베이스라인 1: 최저 인덱스 채널 + 동결 ASR + 균등 방위
baselines/dsp_srp_phat.py    베이스라인 2: GCC-PHAT / SRP-PHAT
baselines/__init__.py        두 베이스라인이 공유하는 로더·그리드·제출 레코드 작성기
tests/test_evaluate.py       평가기 유닛테스트 (make test 가 discover 한다)
requirements.txt             의존성 (최소 / 전체 두 단)
Makefile                     help, compile, test, selftest, sim-selftest, eval-dev, baseline, score, clean
```

**하드코딩 금지.** `36`, `0.25`, `180.0`, `5.0`, `1e-12` 같은 수를 파일에 직접 쓰지 마라.
`from spec import ...` 또는 `import spec`으로 가져와라. 계약값이 두 곳에 적히는 순간 두
값은 반드시 갈라진다.

무거운 의존성(torch, torchaudio, transformers, pyroomacoustics)은 **함수 안에서 지연
임포트**한다. 모듈 최상단에서 import하지 않는다. 그래야 `make compile`과 `make test`가
numpy만으로 항상 통과한다.

---

## 커밋 규약

영어 **Conventional Commits**를 쓴다.

```
feat:  새 기능 (평가기 지표 추가, 베이스라인 추가)
fix:   버그 수정
test:  테스트 추가·수정
docs:  문서만 변경
```

예시:

```
feat(evaluate): add HPD@90 interval width for distance bins
fix(simulate): keep every source on PLANE_HEIGHT_M so 2D distance is the true distance
test(spec): cover azimuth wrap at +180 degrees
docs(readme): declare the 500MB eval-dev ceiling with the per-session arithmetic
```

`spec.py`를 건드리는 커밋은 **반드시** `SPEC_VERSION`을 올리고 `SPEC_DECISIONS.md`를 같이
수정한다. 그 커밋은 단독 PR로 올리고 다른 변경을 섞지 않는다 -- 그 한 줄이 기존 데이터와
기존 점수를 전부 무효화하기 때문이다.
