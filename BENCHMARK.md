# VC-SASR 본 평가 벤치마크 설정

본 평가(2인 화자 귀속)의 고정 조건을 한곳에 모은 문서다. 모든 시스템은 아래 조건을 똑같이 쓰고, 시스템 사이 차이는 화자 임베딩에 무엇을 넣었는지에서만 나온다. 자세한 근거와 재현 명령은 [README](README.md), 채점 계약은 [`spec.py`](spec.py)와 [`SPEC_DECISIONS.md`](SPEC_DECISIONS.md)가 기준이다. 이 문서와 코드가 다르면 코드가 맞다.

## 바로 실행

| 노트북 | 하는 일 | 런타임 |
|---|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hd1534/variable-channel-spatial-asr/blob/main/notebooks/train_template.ipynb) `train_template.ipynb` | 학습·검증: 비교군 3종의 임계값, 제안 모델 3종과 Track A 2종의 학습과 최고 체크포인트 (`CONFIG="all"`) | GPU (T4 가능, A100 권장) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hd1534/variable-channel-spatial-asr/blob/main/notebooks/benchmark.ipynb) `benchmark.ipynb` | 본 평가: 학습한 시스템 6종을 두 평가셋·두 전사기로 채점하고, 최신 단일채널 비교군 3종과 Track B도 같은 채점기로 잰다 (`SYSTEMS="all"`, Track B만은 `"track_b"`) | GPU. 단일채널 비교군은 A100 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hd1534/variable-channel-spatial-asr/blob/main/notebooks/track_c.ipynb) `track_c.ipynb` | Track C(공간 통합 ASR): 단어 정렬, 학습·검증, 본 평가 채점까지 한 노트북에서 (`full_audio` 참고 행) | GPU (A100 권장) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hd1534/variable-channel-spatial-asr/blob/main/notebooks/reproduce.ipynb) `reproduce.ipynb` | 사전 측정(단일 화자) Baseline 2 추론·채점 재현 | CPU |

## 1. 평가셋

| 항목 | 값 |
|---|---|
| 원천 음성 | `LibriSpeech` `test-clean` 화자 40명(남녀 20/20). 학습·검증 코퍼스에 이 화자가 섞이면 학습 코드가 시작을 거부한다 |
| 세션 | 200개, 세션당 20.0초, 세션마다 새 방, 화자 2명(가까운 쪽 0.5\~1.9 m, 먼 쪽 1.9\~5.0 m) |
| 채널 마스크 | `1111`(M = 4), `1110`(M = 3), `1100`·`1010`(M = 2), `1000`·`0010`(M = 1) → 조건마다 1,200케이스 |
| 턴 | 2·3·4턴이 67·67·66세션, 화자 교대 1·2·3번이 134·55·11세션, 성별 쌍 남녀 102·여여 49·남남 49세션 |
| 주 조건 `data/eval-t2` | 화자마다 위치 사본 3개(0.1\~0.3 m 이동), 다시 말하는 화자는 다른 사본에서 말한다. 확산 핑크 잡음 SNR 5·10·20 dB |
| 정지 조건 `data/eval-t2-static` | 같은 세션을 첫 위치·무잡음으로. 학습 없는 공간 단서가 이미 상한 근처라 진단용으로 함께 보고한다 |
| 방 음향 | `pyroomacoustics` **0.8.5 고정**. 다른 버전은 RIR이 달라 같은 평가셋이 나오지 않는다 |
| 생성기 | `make_eval_t2.py`. 다시 만든 사본이 커밋된 세트와 바이트 단위로 같다(`make eval-t2-check`) |

## 2. 신호 규격과 관측 범위

| 항목 | 값 |
|---|---|
| 샘플링 | 16 kHz, 16-bit PCM, 마이크 1\~4개(평면 어레이 반경 ≤ 0.12 m) |
| 시간 격자 | 0.25초 간격, 세션당 80시점 |
| 입력 청크 | 시점 `t_k`마다 `[max(0, t_k − 1.0), t_k]` |
| 공식 관측 범위 | `causal`: `t_k` 이후 샘플을 쓰지 않는다. 이전 시점의 상태 누적은 허용한다 |
| 참고 행 | `full_audio`(세션 전체 사용): mono 비교군·Track B. 공식 causal 표와 따로 보고한다 |

## 3. 공통 부품 (통제 변수)

| 역할 | 모델 | 고정 방식 |
|---|---|---|
| 전사기 (주) | [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | revision `7278e1e7…`, `qwen-asr==0.0.6`, 영어 고정, greedy(beam 1), `max_new_tokens=512`. CUDA에서는 GPU 종류와 관계없이 **FP16**이라 T4·A100에서 전사 조건이 같다. 별도 venv에서 실행한다 |
| 전사기 (이전 기준) | [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) | 사전 측정·초기 실측과 비교하려고 함께 채점한다 |
| 목소리 임베딩 | [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (ECAPA-TDNN) | 동결. 기준 채널(가장 낮은 번호의 활성 채널)에서 청크마다 계산한다 |
| 공간 특징 (무학습) | SRP-PHAT 36구간 방위 분포 | Baseline 2와 같은 함수 |
| 화자 슬롯 | 온라인 코사인 슬롯 매칭 `models/slots.py` | 모든 시스템 공통. 슬롯 상한 2(채점 계약) |

Qwen3-ASR 자체는 화자를 구분하지 않는다. 각 시스템이 예측한 화자 슬롯으로 전사 구간을 정하고, 그 구간 텍스트를 해당 슬롯에 붙인다.

- **전사 구간**: 같은 슬롯이 이어지는 0.25초 격자 구간 하나가 전사 한 번이다. 1.0초(`spec.ASR_MIN_SEGMENT_SEC`)보다 짧으면 양쪽으로 늘린다. 에너지 VAD 구간은 쓰지 않는다. SNR 5·10 dB 세션 134개 중 133개가 20초 한 덩어리로 나와 귀속과 관계없이 모든 단어가 한 슬롯에 가기 때문이다.
- **전사 캐시**: `(평가셋, 세션, 채널, 시작, 끝)`을 키로 모든 시스템이 공유한다. 같은 구간은 한 번만 전사한다.

## 4. 비교 시스템

| 설정 | 시스템 | 학습 | 본 평가 변형 |
|---|---|---|---|
| `voice_only` | 동결 ECAPA + 온라인 군집 | 임계값만(검증 세트, M별) | `full` |
| `space_only` | SRP-PHAT 방위 분포 + 온라인 군집 | 임계값만 | `full` |
| `naive_fusion` | 두 유사도를 M별 고정 가중치로 합산 | 가중치·임계값만 | `full` |
| `proposed_carry` | TAC 공간 경로 + 동결 ECAPA + 2층 인과 GRU 게이트, 청크 간 상태 유지 | 학습 | `full`, `learned_space`, `fixed_gate` |
| `proposed_reset` | 같은 구조, 청크마다 상태 초기화 | 학습 | 같음 |
| `proposed_carry_nodrop` | `proposed_carry`를 4채널 고정으로 학습한 대조군(채널 드롭 없음) | 학습 | 같음 |
| `track_a` | Track A: 사전학습 모델 없이 Log-Mel + 마이크 쌍 GCC-PHAT를 받는 경량 인과 Conformer(약 2.3M) | 학습 | `full`(causal), `full_context`(`full_audio` 참고 행) |
| `track_a_spatial` | `track_a`에서 Log-Mel을 뺀 대조군(위치 정보만) | 학습 | 같음 |

- **제안 모델 학습**: 공간 경로는 화자 손실이 아니라 음원 위치 식별 손실로만 학습한다. 채널 간 특징만으로도 화자가 약 0.88 맞아서 생기는 누수를 막기 위해서다. 게이트는 목소리를 입력받지 않고, M = 1에서는 구조상 0이라 목소리 전용과 같다.
- **학습 기본값**: 20,000걸음, 배치 16, AdamW lr 1e-3, warm-up 500, 혼합 정밀도.
  - 학습 화자: `train-clean-100` 251명
  - 학습 RIR: 방 800개
  - 학습 중 채널 수 비중: M = 2·3·4 균등(`m_weights`)
- **진단 변형**: `learned_space`는 학습된 공간 경로만 쓴 것이고, `fixed_gate`는 게이트를 M별 고정 가중치로 바꾼 것이다. 고정 가중치도 게이트와 마찬가지로 M = 1에서는 0이다. carry의 공간 상태는 마이크가 하나여도 세션 내 시각에 따라 달라지는데, 턴이 시간상 이어져 있으므로 시각만으로도 화자가 갈린다(검증 위치 정확도: carry 0.9975, reset 0.46). 이 단서는 공간 정보가 아니므로 공간 이득에서 뺀다.
- **Track A**: 학습 배치마다 인과 모드와 전체 문맥 모드를 무작위로 골라, 한 가중치로 두 관측 범위를 함께 학습한다.
  - M = 1까지 채널을 드롭하고, 체크포인트도 M = 1~4로 고른다.
  - 같은 데이터·같은 예산(20,000걸음)으로 학습한다.
  - 설계와 근거는 [docs/track-a-design.md](docs/track-a-design.md)에 있다.
- **검증 세트**: 화자는 `dev-clean`에서 사전 측정 화자 6명을 뺀 34명이다. 본 평가 주 조건과 같은 꼴로 200세션을 만든다. 임계값·가중치와 최고 체크포인트를 여기서 고르고, 본 평가셋에서는 아무것도 조정하지 않는다.

## 5. 지표

공식 채점기 `evaluate.py --track 2`, 채점 계약 `spec.py` **0.3.0**을 쓴다.

| 지표 | 정의 | 집계 |
|---|---|---|
| `Spk Acc` | 활성 시점의 화자 일치율. 예측·참조 슬롯의 최적 순열은 시점 일치 수로만 고른다(전사가 귀속 점수를 바꾸지 못한다) | 케이스 평균(`macro_per_case`) |
| `cpWER` | 화자별로 이어 붙인 전사의 WER. 순열은 cpWER 최소로 따로 고른다 | 전체 편집거리 / 전체 참조 단어(`micro_global`) |

결과는 M = 4 → 3 → 2 → 1 순서로 보고한다. M = 2와 M = 1은 마스크 두 개의 평균이다.

## 6. 참고 수치

- **causal 오라클 0.948**: 들은 화자를 이어 부른 값이다. 라벨 규칙의 활성 기준으로 "들은 화자"를 정한 근사라 엄밀한 상한은 아니다.
- **학습 없는 비교군 (`Spk Acc`)**: 아래 표는 공식 채점기 값이다(`runs/eval/*-evaltuned`, `*-statictuned`).
  - 임계값을 각 조건의 평가셋에서 M별로 골랐으므로 낙관적인 값이다. 본 결과는 검증 세트 임계값으로 다시 잰다.
  - 전사기는 wav2vec2다.

| 조건 | 목소리 전용 | 공간 전용 M = 4 / 3 / 2 / 1 | 단순 결합 M = 4 / 3 / 2 / 1 |
|---|---|---|---|
| 주 조건 | 0.833 | 0.878 / 0.853 / 0.803 / 0.659 | 0.878 / 0.859 / 0.833 / 0.833 |
| 정지 조건 | 0.894 (M = 1은 0.891) | 0.963 / 0.950 / 0.935 / 0.659 | 0.964 / 0.954 / 0.935 / 0.891 |

## 7. 본 결과 (측정 중)

`train_template.ipynb`(`CONFIG="all"`)로 학습·검증한 뒤 `benchmark.ipynb`(`SYSTEMS="all"`)로 채점한 결과로 채운다. 원본은 다음 경로에 있다.
- 시스템별 평가: `eval/<시스템>-<전사기>/<평가셋>/<변형>/score.json`
- 학습 요약: `runs/<시스템>/summary.json`. 쓴 GPU와 혼합 정밀도도 여기에 기록된다.

| 시스템 | 전사기 | Spk Acc M = 4 / 3 / 2 / 1 | cpWER M = 4 / 3 / 2 / 1 |
|---|---|---|---|
| 목소리 전용 | Qwen3-ASR-1.7B | — | — |
| 공간 전용 | Qwen3-ASR-1.7B | — | — |
| 단순 결합 | Qwen3-ASR-1.7B | — | — |
| 제안 (Carry-Over) | Qwen3-ASR-1.7B | — | — |
| 제안 (Chunk-Reset) | Qwen3-ASR-1.7B | — | — |
| 제안 (드롭 없음) | Qwen3-ASR-1.7B | — | — |

wav2vec2 전사 결과와 정지 조건 결과도 같은 꼴로 덧붙인다. mono 비교군과 Track B는 `full_audio` 참고 행으로 따로 둔다.

**Track C(공간 통합 ASR)**: Qwen3-ASR-1.7B에 마이크 쌍 공간 어댑터·LoRA·0.25초 판정 층을 붙여, 한 모델이 화자 표시가 붙은 전사문과 시점별 화자를 함께 낸다.
- `track_c`와 대조군 `track_c_mono`(공간 어댑터 없음)를 같은 데이터·같은 예산으로 학습해 공간의 몫을 나눠 본다.
- 20초 전체를 보므로 `full_audio` 참고 행이다.
- 설계는 [docs/unified-asr-design.md](docs/unified-asr-design.md)에 있다.

**Track A(초경량 Conformer)**: 코드는 있고 학습 전이다. `train_template.ipynb`(`CONFIG="all"`에 포함, 따로 돌릴 때는 `"all_track_a"`)로 학습하고 `benchmark.ipynb`(`SYSTEMS="all_track_a"`)로 채점한다.
