# VC-SASR — Variable-Channel Spatial ASR Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hd1534/variable-channel-spatial-asr/blob/main/notebooks/reproduce.ipynb)

**1~4개 동기 마이크 입력을 단일 가중치로 처리하여, 발화 전사와 세션 내 익명 화자 구분, 방위각·거리의 이산 확률 분포 및 대표 위치를 함께 출력하는 공간 음성인식 벤치마크.** 실행 시점에 활성 마이크 수가 달라지더라도 별도의 채널별 체크포인트로 교체하지 않고 하나의 모델로 처리하는 것을 전제로 한다.

분포 출력을 함께 요구하는 이유는 배열의 물리적 크기 때문이다. 평면 어레이 반경 상한이 0.12 m(최대 마이크 쌍 간격 0.24 m)이므로 공간 에일리어싱 한계 주파수가 `spec.ALIASING_LIMIT_HZ` = **714.6 Hz**에 형성된다. 음성 에너지의 상당 부분이 이 주파수 위에 분포하므로 소구경 배열에서는 협대역 위상차만으로 방위를 유일하게 결정하기 어렵다. 특히 활성 마이크가 2개인 조건(`M = 2`)에서는 배열 축에 대한 대칭성 때문에 방위각 사후분포가 본질적으로 **양봉(bimodal)** 형태를 띤다. 단일 좌표 오차만으로는 이러한 기하학적 불확실성을 모델이 제대로 포착했는지 구분할 수 없으므로, 방위 36-bin·거리 8-bin 확률 분포를 제출받아 NLL과 90% HPD 구간으로 함께 채점한다.

---

## 구현 및 실측 현황 (2026-10-29 기준)

평가 계약(`spec.py`), 공식 채점기(`evaluate.py`), 공개 평가셋(`eval-dev` 864케이스), 무학습 통제 베이스라인 2종의 전수 추론 및 채점까지 완료되어 저장소에 반영되어 있다. 제안 학습 모델(Recipe 1, TAC + 인과 GRU)은 2026-12-07 기한에 맞춰 학습을 진행할 예정이다.

아래 상태 표의 항목은 모두 실제 실행으로 검증한 결과다.

| 항목 | 상태 |
|---|---|
| 평가 계약 `spec.py` (`SPEC_VERSION 0.2.0`) | 완료 — doctest 및 자체 정합성 검사 통과 |
| 공식 평가기 `evaluate.py` | 완료 — CLI 플래그 `--pred / --gt / --track / --out` |
| 세션 설계·합성 라이브러리 `simulate.py` | 완료 — 실내 음향 합성 및 설계 불변식 자체 점검 |
| `eval-dev` 생성기 `make_eval_dev.py` | 완료 — 디스크에 평가셋을 기록하는 공식 엔트리포인트 |
| 통제 베이스라인 1·2 (무학습 추론) | 완료 — `baselines.single_channel`, `baselines.dsp_srp_phat` |
| `make test` | **통과** — 모듈 doctest 및 `tests/` 단위 테스트 164개 |
| `make selftest` | **통과** — `simulate` 불변식 및 베이스라인 1·2 종단 점검 |
| `make eval-dev` | **통과** — `pyroomacoustics` + `LibriSpeech` 기반 144세션(864케이스) 생성 (`official: true`) |
| `make baseline` | **통과** — 864케이스 추론 완료 (동결 ASR `wav2vec2-base-960h`) |
| `make score` | **통과** — 864케이스 공식 채점 완료 (`results/baseline1.json`, `results/baseline2.json`) |
| **Recipe 1 제안 모델 학습** | **학습 예정** (마감 2026-12-07) |
| **리더보드** | 통제 베이스라인 2종 **실측 반영 완료**, 제안 모델 및 외부 참조 모델은 측정 예정 |
| Track 2 (2인 교대 발화) | 후속 확장 범위 (`spec.py` 채점 규칙만 동결 완료) |
| 실측 RIR(Unseen Room) 일반화 | 후속 확장 범위 |

현재 저장소와 웹 리더보드에 수록된 베이스라인 수치는 모두 `make eval-dev` → `make baseline` → `make score` 파이프라인으로 864케이스 전체에서 측정한 값이다. 공식 채점기가 `breakdown.by_n_channels` 및 `breakdown.by_mask` 분해 통계를 함께 산출하므로 채널 수별·마스크별 보조 표 역시 수작업 집계 없이 채점기 출력에서 직접 생성된다.

후속 단계에서 채워질 항목은 다음과 같다.

- **Recipe 1 (TAC + GRU)**: 제안 학습 모델. 동일 GRU 구조 위에서 청크마다 은닉 상태를 초기화하는 설정(`Chunk-Reset`)과 유지하는 설정(`Carry-Over`)을 비교하며, 리더보드의 `Ours` 두 행에 해당한다.
- **외부 비통제 참조 모델 (`Qwen3-ASR-1.7B`)**: 단일 채널 전용 대규모 ASR을 구동해 전사 상한 참고치로만 별도 표기한다. 통제 베이스라인과 음향 백엔드가 다르므로 본 순위 경쟁에서는 분리한다.
- **Track 2**: 2인 비중첩 교대 발화 조건. `spec.MAX_SPEAKERS_PER_TRACK[2] = 2` 및 `MULTI_SPEAKER_WER_CONCAT_ORDER` 규칙은 `spec.py`에 정의되어 있으며, 합성 생성기와 화자 추적 베이스라인은 후속 증분으로 분리했다.

남아 있는 스키마 보완 사항 두 건(`meta.jsonl`의 케이스별 행 필드 상수화 및 평가셋 디렉터리 파일명 상수화)은 `SPEC_DECISIONS.md` 14절에 정리되어 있으며, 제출 파일 채점 결과에는 영향을 주지 않는다. 또한 통제 베이스라인 1·2는 모두 별도 학습이 필요 없는 추론 전용 구성이므로 GPU 없이도 전체 파이프라인을 재현할 수 있다(동결 ASR 체크포인트 최초 1회 다운로드만 필요).

---

## 연구 일정 (모델 학습 마감: **2026-12-07**)

일정 지연 시 핵심 산출물이 미완성으로 남지 않도록 주차별 목표와 축소 우선순위를 함께 정의한다.

| 기간 | 일수 | 목표 산출물 | 일정 지연 시 축소 기준 |
|---|---|---|---|
| **10/29 (목) ~ 11/03 (화)** | 6 | 평가 계약 동결, `evaluate.py`·`simulate.py` 구현, `eval-dev` 생성, 베이스라인 1·2 실측 | 세션 수를 줄이되 **고정 마스크 6종은 유지**한다. 마스크 구성을 줄이면 채널 수 변화에 따른 분포 전이를 검증할 수 없기 때문이다. |
| **11/04 (수) ~ 11/17 (화)** | 14 | Recipe 1(가변 채널 단일 가중치) 학습, `causal` 추론 경로 구축 및 예측 파일 생성 | 추가 아키텍처 변형(Recipe 2·3)을 제외하고 Recipe 1 단일 모델 실험에 집중한다. |
| **11/18 (수) ~ 11/30 (월)** | 13 | 절제 실험: Exp 1(채널 수 `M = 1..4` 효과), Exp 2(청크 간 상태 전달 여부 비교) | 일정 부족 시 부가 실험인 Exp 2보다 벤치마크 핵심 검증인 Exp 1을 우선 확보한다. |
| **12/01 (화) ~ 12/07 (월)** | 7 | 최종 리더보드 갱신, HTML 페이지·저장소·PR 정리 | 실측 RIR 일반화 및 `windowed`·`full_audio` 보조 표보다 `causal` × 마스크 6종 주 리더보드 완결을 우선한다. |

---

## 최소 성립 구성 (MVS)

벤치마크 웹페이지, 공식 리더보드, 평가 저장소가 독립적으로 성립하기 위한 최소 단위 구성은 다음과 같다.

```text
Track 1 (단일 정지 화자)
  × observation_mode = causal          (spec.OFFICIAL_SCOPES)
  × spec.EVAL_MASKS 6종                (1000 / 0010 / 1100 / 1010 / 1110 / 1111)
  × Controlled Baseline 1, Baseline 2  (무학습 추론)
  = 공식 주 리더보드 1개 + spec.OFFICIAL_METRICS 15개 중 Track 1 정의 지표 14개
    (cpwer는 spec.TRACK_UNDEFINED_METRICS에 따라 규칙상 N/A)
```

케이스 식별자는 `session_id × mask`(세션당 6개)로 정의되며, 20초 세션에 0.25초 시간 격자를 적용하므로 케이스당 시점 레코드 80개(세션당 공간 레코드 480개)가 생성된다. 이 구성만으로 NLL, HPD 커버리지·구간 폭, 방위·거리 MAE, 허용 오차 정확도, WER이 모두 완결되게 계산된다.

아래 항목은 기본 구성을 확장하는 후속 증분으로 분류한다.

- Track 2 (2인 교대 발화 — `cpwer`가 처음 의미를 갖는 조건)
- 실측 RIR(Unseen Room) 일반화 평가
- 추가 프론트엔드 변형(Recipe 2·3)
- `windowed` / `full_audio` 관측 범위 보조 리더보드 (`windowed`는 미래 참조 창 `W`가 단일 값으로 고정되지 않으므로 비공식 참고 표로만 분리)

---

## 5분 재현 안내

```bash
git clone https://github.com/hd1534/variable-channel-spatial-asr.git vcsasr
cd vcsasr

# 1) 제출 파일 채점만 재실행할 경우 (최소 의존성)
python3 -m pip install numpy==2.2.4

# 2) 데이터셋 생성 및 베이스라인 추론까지 전체 실행할 경우
python3 -m pip install -r requirements.txt

# 3) 평가 계약 및 단위 테스트 검증 (대형 모델·네트워크 불필요)
make test
make selftest

# 4) 공식 평가셋 생성 -> 베이스라인 추론 -> 채점
make eval-dev          # data/eval-dev/ 아래 audio/, ground_truth.jsonl, meta.jsonl 생성
make baseline          # predictions.jsonl 생성
make score             # scores.json 산출
```

공식 `make eval-dev` 실행에는 `LibriSpeech` 원본과 `pyroomacoustics`가 필요하다. 두 자원이 없는 오프라인 환경에서 파이프라인 동작만 빠르게 점검하려면 아래와 같이 내장 합성 모드를 사용할 수 있다. 단, 합성 스모크 모드의 산출물에는 `official: false`가 기록되며 공식 벤치마크 보고에는 사용하지 않는다.

```bash
make eval-dev N_SESSIONS=4 EVAL_DEV_ARGS="--synthetic --rir-engine fallback_ism"
make baseline BASELINE_ARGS="--synthetic-asr"
make eval-dev-verify N_SESSIONS=4   # 오디오 렌더링 없이 설계 불변식만 검증 (numpy 불필요)
```

주요 검증 및 실행 타깃 요약:

```bash
make test              # doctest 5개 모듈 + tests/ 단위 테스트 164개
make selftest          # 합성 세션 기반 베이스라인 1·2 종단 자체 점검
make sim-selftest      # simulate.py 설계 계층 불변식 검증
make eval-dev N_SESSIONS=4 EVAL_DEV_ARGS="--synthetic --rir-engine fallback_ism"
make score             # 제출 파일 채점 (결측 레코드 포함 시 R-MISS 페널티 적용)
```

`Makefile`의 주요 변수(`TRACK`, `SCOPE`, `BASELINE`, `N_SESSIONS`, `SEED`, `DATA_DIR`, `GT`, `PRED`, `SCORES`, `SIM_SESSIONS`)는 명령행에서 덮어쓸 수 있으며, 전체 타깃 설명은 `make help`로 확인할 수 있다.

> **CLI 플래그 및 인터페이스 정합성 메모** (`--help` 기준 검증 완료):
>
> - `make baseline`: 베이스라인 1·2 모두 `--eval-dir / --out / --scope / --track` 인자를 받는다. `BASELINE=2`의 모듈 경로는 `baselines.dsp_srp_phat`이다.
> - `make score`: `evaluate.py`의 인자는 `--pred / --gt / --track / --out / --indent`다. 정답 파일 인자는 `--gt`이며 별도의 `--scope` 플래그는 두지 않는다. 평가기는 제출 레코드 내부의 `observation_mode` 필드를 읽어 `spec.OFFICIAL_SCOPES` 기준으로 공식·비공식 집계를 직접 분리한다.
> - `make eval-dev`: 평가셋 생성 엔트리포인트는 `make_eval_dev.py`(`--out / --n-sessions / --seed / --synthetic / --rir-engine / --verify-only`)이며, `simulate.py`는 음향 합성 및 자체 점검용 라이브러리다.
> - `MANIFEST`: 베이스라인 실행 시 `--manifest $(DATA_DIR)/meta.jsonl`을 명시적으로 전달한다. 디렉터리 내 요약 파일인 `manifest.json`과 케이스별 입력 목록인 `meta.jsonl`을 혼동하지 않도록 분리하기 위함이다.
> - `SCOPE` 기본값은 문자열 하드코딩 대신 `spec.OFFICIAL_SCOPES[0]`에서 읽어 온다.

---

## 제출 파일 스키마

제출 파일(`predictions.jsonl`)은 한 줄에 JSON 객체 하나씩 기록하는 **JSONL** 형식이며 두 종류의 레코드로 구성된다. 아래 예시는 `spec.py` 검증을 통과하는 실제 형식(`azimuth_probs` 36개, `distance_probs` 8개, 각 합 1.0)이다.

### 레코드 1: `timepoint` — 격자 시점별 공간·화자 예측

케이스당 80개(세션당 6개 마스크 합계 480개)를 기록한다. 가독성을 위해 줄바꿈해 표시했으나 실제 JSONL 파일에서는 레코드 전체가 한 줄로 직렬화된다.

```json
{
  "record_type": "timepoint",
  "case_id": "sess_0001__m1010",
  "eval_time_sec": 7.25,
  "pred_speaker": "s0",
  "azimuth_probs": [0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.00031, 0.000757, 0.006822, 0.043523, 0.134681, 0.196431, 0.134681, 0.043534, 0.00717, 0.005653, 0.032731, 0.101084, 0.147394, 0.101084, 0.032716, 0.00519, 0.000641, 0.000306, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294, 0.000294],
  "distance_probs": [0.023962, 0.095004, 0.226493, 0.303436, 0.226493, 0.095004, 0.023962, 0.005646],
  "pred_azimuth_deg": -36.1,
  "pred_distance_m": 1.79
}
```

위 예시에서 마스크 `1010`은 채널 0과 채널 2만 사용하는 2채널(`M = 2`) 조건이다. 전후방 대칭성으로 인해 방위 분포가 구간 14(중심 −35°)와 구간 21(중심 +35°) 두 봉우리로 갈라지며, `spec.INTERVAL_RULE`에 따른 90% HPD 구간 집합은 `{12, 13, 14, 15, 16, 20, 21, 22}`(비연속 구간, 총 폭 80°, 누적 확률 0.9024)가 된다. 실제 방위각이 −37.4°(구간 14)일 때 커버리지는 1, NLL은 1.6274로 계산된다.

### 레코드 2: `final_transcripts` — 케이스별 최종 화자 전사

케이스당 1개(세션당 6개)를 기록한다.

```json
{
  "record_type": "final_transcripts",
  "case_id": "sess_0001__m1010",
  "observation_mode": "causal",
  "final_transcripts": {"s0": "the quick brown fox jumps over the lazy dog"},
  "rtf": 0.42
}
```

### 필드 명세

`timepoint` 레코드:

| 필드명 | 타입 | 필수 여부 | 단위 | 연관 지표 | 생략·누락 시 처리 |
|---|---|---|---|---|---|
| `record_type` | `string` (`"timepoint"`) | 권장 | - | - | `eval_time_sec` 유무로 추론하며 모호할 경우 거부 (`"timestep"` 하위 호환 별칭 읽기 지원) |
| `case_id` | `string` (`<session_id>__m<mask_bits>`) | 필수 | - | 전 지표 1차 키 | 스키마 오류로 파일 거부 |
| `eval_time_sec` | `float` (`k * spec.DELTA_T_SEC`, 0.25 배수) | 필수 | 초 (`session_start` 기준) | 공간·화자 지표 2차 키 | 스키마 오류로 파일 거부 |
| `pred_speaker` | `string` (세션 내 익명 화자 슬롯; `speaker_id`, `speaker` 별칭 수용) | 선택 | - | `speaker_attribution_accuracy` | `spec.PENALTY_SPEAKER_CORRECT = False` (오답 처리) |
| `azimuth_probs` | `array[36]` of `float` | 선택 | 확률 (합 `1 ± 1e-4`) | `azimuth_nll`, `azimuth_coverage_90`, `azimuth_interval_width_deg` | 레코드 결측 시 `PENALTY_NLL`(27.631) 부과; 점 전용 제출 시 분포 지표 `N/A` |
| `distance_probs` | `array[8]` of `float` | 선택 | 확률 (합 `1 ± 1e-4`) | `distance_nll`, `distance_coverage_90`, `distance_interval_width_m` | 레코드 결측 시 `PENALTY_NLL` 부과; 점 전용 제출 시 분포 지표 `N/A` |
| `pred_azimuth_deg` | `float` (`[-180, 180)`) | 선택 | 도 (°) | `azimuth_mae_deg`, `azimuth_acc_10deg`, `azimuth_acc_20deg`, `pos_error_2d_m` | 생략 시 `azimuth_probs` 최빈(MAP) 구간의 `AZIMUTH_BIN_CENTERS_DEG` 사용 (`R-PT`) |
| `pred_distance_m` | `float` (`[0.5, 5.0]`) | 선택 | m | `distance_mae_m`, `pos_error_2d_m` | 생략 시 `distance_probs` 최빈(MAP) 구간의 기하평균 중심 `DISTANCE_BIN_CENTERS_M` 사용 (`R-PT`) |

> 2차원 직교좌표(`pred_x_m`, `pred_y_m`)도 선택 필드로 함께 또는 단독 제출할 수 있으며, 극좌표와 직교좌표를 동시에 기재한 경우 변환 정합성을 검사한다(`R-POS`).

`final_transcripts` 레코드:

| 필드명 | 타입 | 필수 여부 | 단위 | 연관 지표 | 생략·누락 시 처리 |
|---|---|---|---|---|---|
| `record_type` | `string` (`"final_transcripts"`) | 권장 | - | - | `final_transcripts` 키 유무로 추론 |
| `case_id` | `string` | 필수 | - | `wer`, `cpwer` 키 | 스키마 오류로 파일 거부 |
| `observation_mode` | `string` (`causal`, `windowed`, `full_audio`) | 필수 | - | 관측 범위별 분리 집계 | 스키마 오류로 파일 거부 |
| `final_transcripts` | `object` (`{speaker_id: string}`, 슬롯 수 ≤ 트랙 상한) | 필수 | - | `wer` (`micro_global`), `cpwer` | 케이스 누락 시 참조 단어 전체를 삭제 오류로 계산 (`count_all_deletions`) |
| `rtf` | `float` (추론 시간 / 오디오 길이) | 권장 | 배수 | `rtf` | 미기재 시 `rtf` 열을 `N/A`로 표기 |

모든 정식 필드명과 별칭 해석 순서는 `spec.py`(`F_*` 상수, `FIELD_ALIASES`, `REQUIRED_FIELDS`, `PROB_FIELD_LENGTHS`)에 단일 정의되어 있다(세부 결정 근거는 `SPEC_DECISIONS.md` 13절 참조).

---

## 채점 규칙 요약

`evaluate.py`가 발생시키는 검증 예외 메시지는 아래 규칙 ID를 접두사로 포함한다(예: `R-NLL: azimuth_probs sums to 0.912, tolerance 1e-04`).

| 규칙 ID | 요약 | 관련 `spec.py` 상수 |
|---|---|---|
| **R-DUP** | `(case_id, eval_time_sec)` 또는 케이스별 최종 전사 레코드가 중복되면 임의의 행을 선택하지 않고 제출 파일 전체를 거부한다. | `DELTA_T_SEC`, `DUPLICATE_RECORD_POLICY` |
| **R-MISS** | 활성 평가 시점 결측 시 레코드를 제외하지 않고 최악 페널티(방위 180.0°, 거리 5.0 m, 2D 오차 10.0 m, 화자 오답, NLL 27.631 nats)를 부과한다. 전사 누락 시에는 참조 단어를 전량 삭제 오류로 계산해 전략적 기권 이득을 차단한다. | `PENALTY_AZIMUTH_DEG`, `PENALTY_DISTANCE_M`, `PENALTY_POS_ERROR_2D_M`, `PENALTY_NLL`, `MISSING_TRANSCRIPT_POLICY` |
| **R-PERM** | 예측·참조 화자 슬롯을 `max(len(pred), len(ref))` 크기로 빈 문자열 패딩한 뒤 1:1 최적 순열을 찾는다. 동률은 `cpWER 최소 → 시점별 화자 일치 수 최대 → 참조 슬롯 인덱스 사전순`으로 해소한다. Track 1의 `cpwer`는 규칙상 `N/A`다. | `SPEAKER_PAD_TOKEN`, `PERMUTATION_TIE_BREAK`, `MAX_SPEAKERS_PER_TRACK`, `TRACK_UNDEFINED_METRICS` |
| **R-NLL** | 확률 벡터 길이는 정확히 36 / 8이어야 하며 합이 `1 ± 1e-4`를 벗어나면 재정규화하지 않고 거부한다. 수치 안정성을 위한 `1e-12` 하한 클리핑은 정답 구간 확률에만 적용한다. | `K_AZIMUTH`, `K_DISTANCE`, `PROB_SUM_TOLERANCE`, `NORMALISE_SUBMITTED_PROBS`, `NLL_EPSILON` |
| **R-PT** | 명시적 점 예측이 있으면 우선 사용하고, 생략된 경우에만 분포의 MAP 구간 대표점(방위 산술 중심, 거리 기하 중심)으로 점 지표를 계산한다. | `AZIMUTH_BIN_CENTERS_DEG`, `DISTANCE_BIN_CENTERS_M` |
| **R-TXT** | 참조와 가설 텍스트 모두에 `NFKC → 소문자화(casefold) → 구두점 처리 → 숫자 영단어 변환(0~999) → 공백 정리`를 동일 순서로 적용한다. | `TEXT_NORMALISATION`, `WER_AGGREGATION`, `SPATIAL_AGGREGATION` |

집계 방식 구분:

- **공간·화자 지표 (`macro_per_case`)**: 각 케이스 내 활성 발화 시점(`T_eval`)에 대해 균등 평균을 낸 뒤, 전체 케이스에 동일 가중치를 부여해 평균한다. 발화 길이가 긴 특정 세션이 전체 공간 점수를 지배하지 않도록 하기 위함이다.
- **전사 오류율 (`micro_global`)**: 표준 ASR 관례에 따라 전체 케이스의 편집거리 총합을 전체 참조 단어 수로 나눈다.
- **불확실성 구간 (`hpd_greedy`)**: 확률 내림차순으로 구간을 선택해 누적 확률이 0.90에 도달하는 최소 집합을 구성한다. 비연속 구간을 허용하므로 `M = 2` 양봉 분포의 두 후보 방향이 중간 빈 구간 없이 정확히 반영된다.

---

## eval-dev 배포 규격

### 오디오 및 라벨 용량 산정

4채널 · 16 kHz · 16-bit PCM(`pcm_s16le`) 기준 초당 데이터양은 다음과 같다.

```text
128,000 B/s = 4 ch × 16,000 Hz × 2 B  (= 128 kB/s, 125 KiB/s)
```

세션 길이는 `spec.SESSION_LENGTH_SEC = 20.0`초로 고정되므로 세션 수에 따른 용량은 아래와 같다.

| 항목 | 계산식 | 용량 |
|---|---|---|
| 세션 1개 오디오 (4ch WAV) | `128,000 × 20` | **2.56 MB** (2.441 MiB) |
| 세션 1개 라벨 (80시점 기준) | 격자 80점 | 약 **13.0 kB** |
| 세션 100개 | 오디오 기준 | 256.0 MB |
| **공식 배포본 (144세션)** | 오디오 351.6 MB + 라벨·메타 약 6.1 MB | 약 **358 MB** |
| 기본 `N_SESSIONS = 150` 상한 예시 | 오디오 384.0 MB + 라벨 약 2.0 MB | **386.0 MB** |

오디오 파일은 마스크별로 중복 저장하지 않고, 4채널 세션을 1회 저장한 뒤 로더가 `spec.mask_to_channels`로 활성 채널만 추출한다(마스크 6종의 활성 채널 합은 13개이므로 개별 저장 시 용량이 3.25배 증가함). 추론 결과인 `predictions.jsonl`은 확률 벡터를 포함해 세션당 약 303 kB(150세션 기준 약 45.5 MB)이므로 `.gitignore`로 버전 관리에서 제외한다.

### 저장소 용량 기준

| 구분 | 기준값 | 설정 근거 |
|---|---|---|
| **`eval-dev` 전체 용량** | **≤ 500 MB** | GitHub 단일 저장소 권장 규모(1 GB) 이내로 여유 공간 확보 |
| **단일 파일 크기** | **≤ 90 MB** | GitHub 파일당 업로드 제한(100 MB) 미만의 안전 한계 |
| **공식 커밋 규모** | **약 358 MB** | 144세션 개별 WAV(2.44 MiB) + JSONL 라벨 및 해시 매니페스트 |

### 커밋 포함 및 제외 대상

챌린지 제출 요건(개인 저장소 내 평가용 데이터셋 직접 포함)에 맞춰 `data/eval-dev/`의 평가 입력과 정답을 저장소에 함께 포함한다.

| 경로 | 설명 | Git 추적 여부 |
|---|---|---|
| `data/eval-dev/audio/` | 144세션 4채널 WAV (총 351.6 MB, 파일당 2.44 MiB) | 포함 |
| `data/eval-dev/ground_truth.jsonl` | 공식 채점용 정답 라벨 (5.43 MB) | 포함 |
| `data/eval-dev/meta.jsonl` | 추론 입력용 케이스별 메타데이터 (0.58 MB) | 포함 |
| `data/eval-dev/manifest.json` | 파일별 SHA-256 해시 및 검증 통계 요약 (0.05 MB) | 포함 |
| `data/eval-dev/ATTRIBUTION.md` | `LibriSpeech` 파생 오디오 출처 및 라이선스 고지 | 포함 |
| `data/eval-dev/rir/` | 합성 중간 산출물(15.0 MB, 동일 시드로 재생성 가능) | 제외 |
| `runs/*.jsonl` | 모델 추론 중간 결과물 | 제외 |

### Git LFS 대신 일반 Git 추적을 선택한 이유

351.6 MB 규모의 오디오 파일은 통상 Git LFS 대상이지만, GitHub 무료 계정의 LFS 월간 대역폭 한도(1 GB/월) 때문에 공개 평가용 저장소에서는 오히려 다운로드 장애를 유발할 수 있다. 저장소를 세 번만 복제(clone)해도 월간 대역폭 쿼터가 소진되어 이후 채점자나 재현자가 오디오를 내려받지 못하게 되기 때문이다.

개별 WAV 파일 크기가 2.44 MiB로 GitHub 단일 파일 제한(100 MB)보다 충분히 작고, 고정 시드(`20261029`) 기반 공식 세트(`official: true`)는 반복 덮어쓰기가 발생하지 않으므로 일반 Git 객체로 직접 추적하여 복제 가용성을 보장했다.

---

## 라이선스 및 출처 표기

### LibriSpeech 파생 오디오 고지

원천 음성 코퍼스인 `LibriSpeech`(`dev-clean`)는 **CC BY 4.0** 라이선스로 배포된다. 실내 임펄스 응답을 합성한 다채널 오디오 역시 파생 저작물에 해당하므로 재배포 시 원저작물 출처와 변경 내역을 함께 명시해야 하며, 상세 고지문은 `data/eval-dev/ATTRIBUTION.md`에 포함되어 있다.

```text
This dataset contains audio derived from LibriSpeech (Panayotov et al., 2015),
distributed under CC BY 4.0. The derived multichannel audio was produced by
simulated room acoustics and is redistributed under the same license.
Source: https://www.openslr.org/12
```

### 외부 데이터셋 · 라이브러리 · 체크포인트 라이선스

| 자원 | 저장소 내 용도 | 라이선스 | 비고 |
|---|---|---|---|
| `LibriSpeech` (`dev-clean`) | 합성 평가셋 원천 음성 | `CC BY 4.0` | 파생 오디오 재배포 시 `ATTRIBUTION.md` 동봉 필수 |
| `pyroomacoustics` | 다채널 실내 임펄스 응답(ISM) 생성 | `MIT` | 라이브러리 코드 미재배포 |
| `facebook/wav2vec2-base-960h` | 통제 베이스라인 1·2 및 제안 모델 동결 ASR 백엔드 | `Apache-2.0` | Hugging Face 런타임 로드 (가중치 미재배포) |
| `openai/whisper` | 모듈형 비교 레시피용 동결 ASR 후보 | `MIT` | 가중치 미재배포 |
| `speechbrain/spkrec-ecapa-voxceleb` | Track 2 확장용 동결 화자 임베딩 백엔드 | `Apache-2.0` | 기본 의존성에서는 분리, 별도 환경 사용 권장 |
| `snakers4/silero-vad` | 모듈형 파이프라인 VAD 후보 | `MIT` | 현재 통제 베이스라인은 내장 에너지 게이트 사용 |
| `transformers` | 사전학습 음향 백엔드 로더 | `Apache-2.0` | - |
| `torch` / `torchaudio` | 텐서 연산 및 모델 추론 | `BSD-3-Clause` | 동일 마이너 버전 조합 필요 |
| `numpy` / `scipy` | 채점기 벡터 연산 및 SRP-PHAT 신호처리 | `BSD-3-Clause` | `evaluate.py`는 `numpy` 미설치 시 표준 라이브러리 폴백 동작 |
| `soundfile` | 다채널 WAV 입출력 바인딩 | `BSD-3-Clause` | 동적 링크되는 `libsndfile`은 `LGPL-2.1+` |
| `pyannote.audio` | 현재 미사용 (후속 비교 후보) | `MIT` (코드) | Hugging Face 사전학습 가중치는 별도 이용 약관 동의(gated) 필요 |

---

## 저장소 디렉터리 구조

```text
spec.py                      동결된 평가 계약 상수 및 스키마 정의 (수치·필드명의 단일 출처)
SPEC_DECISIONS.md            기획안 v1.7의 미정 항목을 spec.py에서 확정한 근거와 파급 범위
evaluate.py                  공식 채점기 (--pred / --gt / --track / --out)
simulate.py                  실내 음향 시뮬레이션 및 층화 설계 라이브러리 (--self-test 지원)
make_eval_dev.py             공식 eval-dev 데이터셋 생성 스크립트 (--out / --n-sessions / --seed)
baselines/single_channel.py  통제 베이스라인 1: 단일 채널 선택 + 동결 ASR + 균일 공간 분포
baselines/dsp_srp_phat.py    통제 베이스라인 2: SRP-PHAT 공간 분포 + 지연-합 빔포밍 + 동결 ASR
baselines/__init__.py        베이스라인 공통 로더, 인과적 청크 격자 및 JSONL 직렬화 유틸리티
tests/test_evaluate.py       평가 계약 및 채점기 단위 테스트 모음 (153개 테스트)
tests/test_baselines.py      베이스라인 대표점 규칙 단위 테스트 (11개 테스트)
notebooks/reproduce.ipynb    무료 CPU 환경에서 동작하는 Colab 재현 노트북
docs/index.html              GitHub Pages 배포용 벤치마크 웹페이지 (제출 HTML과 동일 본)
requirements.txt             최소 채점용 및 전체 파이프라인용 의존성 명세
Makefile                     컴파일, 단위 테스트, 데이터 생성, 베이스라인 추론, 채점 자동화
```

코드 작성 원칙:

- **단일 상수 출처 유지**: `36`, `0.25`, `180.0`, `5.0`, `1e-12` 등 평가 계약과 직결되는 수치는 개별 모듈에 리터럴로 중복 기입하지 않고 `spec.py`에서 임포트해 사용한다.
- **무거운 의존성의 지연 임포트**: `torch`, `torchaudio`, `transformers`, `pyroomacoustics` 등은 필요한 함수 내부에서 지연 임포트(lazy import)하여, 기본 파이썬 환경만으로도 `make compile`과 `make test`가 항상 수행되도록 유지한다.

---

## 커밋 규약

영문 **Conventional Commits** 형식을 따른다.

```text
feat:  새로운 기능 추가 (지표, 베이스라인, 시각화 등)
fix:   버그 및 불일치 수정
test:  단위 테스트 추가·보강
docs:  문서 및 웹페이지 갱신
```

예시:

```text
feat(evaluate): add HPD@90 interval width for distance bins
fix(simulate): keep every source on PLANE_HEIGHT_M so 2D distance matches 3D propagation
test(spec): cover azimuth wraparound at +180 degrees
docs(readme): document eval-dev storage budget and verified licenses
```

`spec.py`의 계약 상수를 변경할 때는 반드시 `SPEC_VERSION`을 상향하고 `SPEC_DECISIONS.md`에 변경 근거와 기존 점수에 미치는 영향을 함께 기록한다.
