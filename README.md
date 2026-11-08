# VC-SASR — Variable-Channel Spatial ASR Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hd1534/variable-channel-spatial-asr/blob/main/notebooks/reproduce.ipynb)

**마이크가 늘어날 때 공간 정보가 음성인식의 화자 구분을 돕는지 검증하는 가변 채널 벤치마크.** 연구 질문은 "마이크의 개수가 늘어난다면, 공간적 특성을 활용하여 기존 ASR의 화자 구분 능력이 향상될까?"이다. 1\~4개 동기 마이크 입력을 단일 가중치로 처리하는 것을 전제로 하며, 실행 시점에 활성 마이크 수가 달라지더라도 별도의 채널별 체크포인트로 교체하지 않고 하나의 모델로 처리한다.

지금의 음성인식이 화자를 가르는 근거는 대부분 목소리 특징(화자 임베딩)이다. 성별과 음색이 비슷한 화자, 짧게 주고받는 발화, 잔향이 큰 실내처럼 목소리 특징이 흐려지는 조건에서는 화자를 혼동하기 쉽고, 말하는 사람이 많아질수록 오류도 커진다. 마이크가 여러 개라면 같은 방 서로 다른 위치에 있는 화자마다 마이크 사이 도달 시간차(TDOA)가 달라, 목소리와 무관한 공간 단서를 더할 수 있다.

마이크 수에 따라 얻는 공간 정보는 배열 기하가 정한다. 평면 어레이 반경 상한이 0.12 m(최대 마이크 쌍 간격 0.24 m)이므로 공간 에일리어싱 한계 주파수가 `spec.ALIASING_LIMIT_HZ` = **714.6 Hz**에 형성되고, 음성 에너지의 상당 부분이 이 주파수 위에 분포하므로 소구경 배열에서는 협대역 위상차만으로 방위를 유일하게 결정하기 어렵다. 마이크 1개에는 방향 단서가 없고, 활성 마이크가 2개인 조건(`M = 2`)에서는 배열 축에 대한 대칭성 때문에 방위각 사후분포가 본질적으로 **양봉(bimodal)** 형태를 띤다. 그래도 화자 구분에는 정확한 방향이 아니라 두 화자의 공간 신호가 다르기만 하면 된다. 본 평가셋 200세션 중 마이크 2개 조건의 78\~86%(마스크 `1100`·`1010`), 4개 조건의 97%에서 두 화자의 도달 시간차가 1샘플 이상 다르므로, 방향 추정은 마이크 3개부터 크게 좋아지지만 화자 구분은 마이크 1개에서 2개로 갈 때 먼저 크게 좋아질 것으로 예상한다.

평가는 두 단계로 이뤄진다.

- **본 평가 (2인)**: 같은 방 서로 다른 위치의 두 화자가 번갈아 말하는 조건에서 공간 단서가 화자 구분을 개선하는지 검증한다. 주 지표는 `cpWER`·`Spk Acc`다. 평가셋은 `make_eval_t2.py`로 만든다.
- **사전 측정 (단일 화자, 측정 완료)**: 화자가 한 명인 조건에서 마이크 수별로 얻을 수 있는 공간 정보의 양을 잰다. 지금 저장소에 있는 평가셋(`eval-dev` 144세션 · 864케이스)과 실측 결과는 모두 사전 측정이다.

방위각 36-bin·거리 8-bin 확률 분포와 대표 위치는 현재 채점 계약(`spec.py` v0.2.0)에서 선택 출력이다. 제출하면 NLL·90% HPD 구간·위치 오차로 채점해 채널 수에 따라 달라지는 방향 불확실성(예: `M = 2`의 앞뒤 두 봉우리)을 진단하지만, 화자 구분 주 지표 계산에는 들어가지 않는다. 거리는 목표가 아니다. 사전 측정셋 144세션의 RIR에서는 거리에 따른 수신 레벨 변화(거리 2배당 약 −1.8 dB)가 방마다 다른 잔향 편차(약 5.8 dB)에 묻히므로, 거리 수치는 사전 측정의 진단값으로만 읽는다.

---

## 구현 및 실측 현황 (2026-10-30 기준)

평가 계약(`spec.py`), 공식 채점기(`evaluate.py`), 사전 측정 평가셋(`eval-dev` 864케이스), 무학습 통제 베이스라인 2종의 전수 추론 및 채점까지 완료되어 저장소에 반영되어 있다. 화자 구분 비교 시스템과 제안 모델(`TAC + ECAPA + GRU`)은 학습·검증 코드와 Colab 템플릿 노트북까지 갖췄고, 실측과 학습(기한 2026-12-07)은 남았다. 본 평가에 쓸 2인 평가셋은 생성기(`make_eval_t2.py`)로 200세션을 만들었다.

아래 상태 표의 완료·통과 항목은 모두 실제 실행으로 검증한 결과다.

| 항목 | 상태 |
|---|---|
| 평가 계약 `spec.py` (`SPEC_VERSION 0.2.0`) | 완료 — doctest 및 자체 정합성 검사 통과. 세션당 화자를 최대 2명까지 채점한다(`MAX_SPEAKERS_PER_TRACK`) |
| 공식 평가기 `evaluate.py` | 완료 — 다화자 `cpWER`·화자 순열 매칭 포함, CLI 플래그 `--pred / --gt / --track / --out` |
| 세션 설계·합성 라이브러리 `simulate.py` | 완료 — 실내 음향 합성 및 설계 불변식 자체 점검 |
| 사전 측정 평가셋 생성기 `make_eval_dev.py` | 완료 — 같은 시드로 다시 만든 오디오·정답·메타가 커밋된 세트와 바이트 단위로 같았다 |
| 사전 측정 평가셋 `data/eval-dev` | 배포 — `pyroomacoustics` + `LibriSpeech` 기반 144세션(864케이스), `official: true` |
| 본 평가셋 `data/eval-t2`, `data/eval-t2-static` | 배포 — `LibriSpeech` `test-clean` 기반 200세션(조건마다 1200케이스), 주 조건·정지 조건 두 벌, `official: true` |
| 통제 베이스라인 1·2 (무학습 추론) | 실측 완료 — `results/baseline1.json`, `results/baseline2.json` (동결 ASR `wav2vec2-base-960h`) |
| `make test` | **통과** — `spec.py` 포함 8개 파일의 doctest 및 `tests/` 단위 테스트 171개 (`torch`가 있으면 학습 코드 테스트 `tests/test_training.py` 54개가 더해진다) |
| `make selftest` | **통과** — `simulate` 불변식 및 베이스라인 1·2 종단 점검 |
| `make smoke` | **통과** — 합성 데이터로 생성 → 추론 → 채점 전 경로 점검 (`runs/smoke` 안에서만) |
| 본 평가셋 생성기 `make_eval_t2.py` | 완료 — `test-clean` 화자 40명, 200세션(1200케이스), 주 조건과 같은 세션의 정지 조건 두 벌. 정답을 예측으로 넣으면 공식 채점기가 두 벌 모두 `Spk Acc` 1.0, `cpWER` 0을 낸다 |
| **화자 구분 비교 시스템** | **코드 완료, 실측 전** — 목소리 전용(ECAPA-TDNN + 온라인 군집), 공간 전용(SRP-PHAT 방위 군집), 단순 결합(두 점수 합산). `models/`, 아래 "화자 구분 시스템 학습" 참조 |
| **제안 모델 `TAC + ECAPA + GRU`** | **학습 코드 완료, 학습 전** (마감 2026-12-07) — 청크 상태 초기화(`Chunk-Reset`)와 유지(`Carry-Over`) 비교. `make train-smoke`·`make train-test` 통과 |

사전 측정 수치는 모두 커밋된 `data/eval-dev`에 베이스라인을 돌리고(`make baseline`) 공식 채점기로 채점한(`make score`) 864케이스 전체 값이다. 공식 채점기가 `breakdown.by_n_channels` 및 `breakdown.by_mask` 분해 통계를 함께 산출하므로 마이크 수별·마스크별 표 역시 수작업 집계 없이 채점기 출력에서 직접 나온다. Baseline 2의 방위 MAE는 마이크 1개 90.21°, 2개 53.12°, 3개 9.51°, 4개 6.77°이며, 전체 표는 웹페이지 5.2\~5.3절에 있다.

남아 있는 스키마 보완 사항 두 건(`meta.jsonl`의 케이스별 행 필드 상수화 및 평가셋 디렉터리 파일명 상수화)은 `SPEC_DECISIONS.md` 14절에 정리되어 있으며, 채점 결과에는 영향을 주지 않는다. 또한 통제 베이스라인 1·2는 모두 별도 학습이 필요 없는 추론 전용 구성이므로 GPU 없이도 사전 측정 전체를 재현할 수 있다(동결 ASR 체크포인트를 `make asr-cache`로 한 번 내려받는 것만 필요).

---

## 남은 작업 (모델 학습 마감: **2026-12-07**)

본 평가에 필요한 순서대로 적는다.

| 순서 | 작업 | 내용 |
|---|---|---|
| 1 | 화자 구분 비교 시스템 실측 | 목소리 전용, 공간 전용, 단순 결합. 코드는 있고(`configs/voice_only.json` 등), 학습 없이 임계값만 검증 세트에서 정한다 |
| 2 | 제안 모델 학습 | `TAC + ECAPA + GRU`, 동일 GRU 구조에서 `Chunk-Reset`과 `Carry-Over` 대조. Colab에서 `notebooks/train_template.ipynb`로 학습한다 |
| 3 | 결과표와 가설 검증 | 웹페이지 5.1절 결과표(마이크 수별 `Spk Acc`·`cpWER`)를 채우고 2.3절 가설 H1\~H4를 판정한다 |

실측 RIR(BUT ReverbDB 등)로 화자 구분 이득이 유지되는지 보는 일반화 점검은 후속 범위로 둔다. 본 평가는 2인으로 정했다. 3인 이상은 학습 검증의 내부 지표로만 보며, 채점 계약(세션당 2인)은 바꾸지 않는다.

---

## 평가 구성

| 구분 | 음향·화자 조건 | 지표 | 상태 |
|---|---|---|---|
| **본 평가 (2인)** | 같은 방 서로 다른 위치의 두 화자가 번갈아 발화(turn-taking). 발화 시점마다 활성 화자는 최대 1명이고, 화자마다 위치 사본(첫 위치에서 0.1\~0.3 m)을 두고 다시 말하는 화자는 다른 사본에서 말하며, 확산 잡음이 깔린다(같은 세션의 정지·무잡음 조건을 짝으로 둔다) | 주 지표 `cpWER`·`Spk Acc` | 평가셋 200세션(1200케이스) 생성. 채점 계약(화자 순열 매칭)은 세션당 2인 |
| **사전 측정 (단일 화자)** | 세션당 단일 정지 화자(1인). 발화 전후 및 어절 사이 침묵 구간 포함, 화자 교대 없음 | 공간 진단 지표(방위·거리 오차, 분포 NLL·HPD), 보조 지표 `WER`·`RTF` | `eval-dev` 864케이스 공개, 베이스라인 2종 실측 완료 |

코드에는 `track`이라는 이름이 남아 있다. `spec.MAX_SPEAKERS_PER_TRACK = {1: 1, 2: 2}`가 세션당 화자 수 상한이고, 채점기와 베이스라인의 `--track` 인자로 고른다. `--track 1`은 단일 화자(사전 측정), `--track 2`는 세션당 2인 채점이다. 본 평가는 2인이므로 이 상한을 그대로 쓴다.

사전 측정의 채점 구성은 다음과 같다.

```text
사전 측정 (단일 정지 화자, --track 1)
  × observation_mode = causal          (spec.OFFICIAL_SCOPES)
  × spec.EVAL_MASKS 6종                (1000 / 0010 / 1100 / 1010 / 1110 / 1111)
  × Controlled Baseline 1, Baseline 2  (무학습 추론)
  = spec.OFFICIAL_METRICS 15개 중 cpwer를 뺀 14개
    (cpwer는 spec.TRACK_UNDEFINED_METRICS에 따라 규칙상 N/A,
     화자가 한 명이라 Spk Acc도 화자 구분 지표로서는 의미가 없다)
```

케이스 식별자는 `session_id × mask`(세션당 6개)로 정의되며, 20초 세션에 0.25초 시간 격자를 적용하므로 케이스당 시점 레코드 80개(세션당 480개)가 생성된다.

관측 범위는 세 가지다. 공식 표는 `causal`이다. 시점 `t_k` 예측에 `t_k` 이후의 샘플을 쓰지 않으며, 매 시점의 입력 청크는 `[max(0, t_k − 1.0), t_k]` 구간이고, 청크 간 은닉 상태 전달(`Carry-Over`)이나 온라인 화자 슬롯처럼 이전 시점까지의 정보를 누적하는 것은 허용한다. 미래 창을 참조하는 `windowed`(제출 레코드에 `lookahead_sec` 필수)와 세션 전체를 쓰는 `full_audio`는 별도 표로 분리하며, `windowed`는 미래 참조 창 `W`가 단일 값으로 고정되지 않으므로 비공식 참고 표로만 둔다.

---

## 재현 안내

사전 측정 평가셋(`data/eval-dev`)이 저장소에 들어 있으므로, 사전 측정 수치를 재현하는 데 평가셋 생성은 필요 없다. 예측 파일은 저장소에 없으므로 베이스라인 추론부터 돌린다.

```bash
git clone https://github.com/hd1534/variable-channel-spatial-asr.git vcsasr
cd vcsasr

# 1) 채점과 단위 테스트만 할 경우 (최소 의존성, 대형 모델·네트워크 불필요)
python3 -m pip install numpy==2.2.4
make test              # spec.py 포함 8개 파일 doctest + tests/ 단위 테스트 171개

# 2) 베이스라인 추론까지 할 경우 (전체 의존성)
python3 -m pip install -r requirements.txt
make selftest          # 합성 세션으로 simulate 불변식과 베이스라인 1·2 종단 점검
make smoke             # 합성 데이터로 생성 -> 추론 -> 채점 (runs/smoke 안에서만 쓴다)
make asr-cache         # 동결 ASR 체크포인트를 한 번 내려받는다 (약 360 MiB, 네트워크 필요)

# 3) 사전 측정 재현: 커밋된 data/eval-dev 에 베이스라인 1·2 추론 -> 채점
make baseline BASELINE=1 PRED=runs/b1.jsonl
make score PRED=runs/b1.jsonl SCORES=runs/b1.scores.json
make baseline BASELINE=2 PRED=runs/b2.jsonl
make score PRED=runs/b2.jsonl SCORES=runs/b2.scores.json
```

3단계 채점 결과는 커밋된 `results/baseline1.json`, `results/baseline2.json`과 비교한다. 이 맥에서 위 명령을 그대로 다시 돌린 결과는 두 베이스라인 모두 `rtf`를 뺀 지표가 커밋된 값과 정확히 같았다(`rtf`는 실행 기기와 부하에 따라 달라진다). 베이스라인은 체크포인트를 스스로 내려받지 않으므로(`HF_HUB_OFFLINE=1`, `local_files_only`) 새 환경에서는 `make asr-cache`를 먼저 돌려야 하며, Colab 노트북은 이 중 Baseline 2의 추론과 채점을 무료 CPU 런타임에서 재현한다.

### 사전 측정 평가셋 다시 만들기 (선택)

`make eval-dev`는 커밋된 `data/eval-dev`가 아니라 `data/eval-dev-rebuild`(git이 무시하는 경로)에 평가셋을 새로 만든다. `LibriSpeech` `dev-clean` 코퍼스(<https://www.openslr.org/12>)와 `pyroomacoustics`가 필요하다.

```bash
make eval-dev-verify   # 오디오 렌더링 없이 설계 불변식만 검증 (numpy 불필요, 아무것도 쓰지 않는다)
make eval-dev LIBRISPEECH_ROOT=<dev-clean.tar.gz 를 푼 디렉터리>
make eval-dev-check    # data/eval-dev-rebuild 의 오디오·정답·메타를 커밋된 data/eval-dev 와 바이트 단위로 비교
```

`LIBRISPEECH_ROOT` 아래에는 `dev-clean`만 있어야 한다. 화자 분할이 루트 아래 화자 전체로 정해지므로, 다른 서브셋이 섞이면 다른 세트가 만들어진다. 세션 수는 `N_SESSIONS = 144`가 기본이다. 방위 36구간·거리 8구간이 엄격히 균일해지는 가장 작은 세션 수이며, `official: true`는 144의 배수에서만 나온다. 정답 파일은 원본 FLAC 경로를 코퍼스 기준 상대 경로(`<화자>/<챕터>/<발화>.flac`)로 기록하므로 코퍼스를 푼 위치와 무관하다. 이 맥(Python 3.13.11, numpy 2.2.4, pyroomacoustics 0.8.5)에서 코퍼스 경로를 바꿔 가며 다시 만든 세트는 모두 오디오 144개와 `ground_truth.jsonl`, `meta.jsonl`이 커밋된 세트와 바이트 단위로 같았다.

생성기(`make_eval_dev.py`)는 `manifest.json`이 `official: true`인 디렉터리에는 `--overwrite-official` 없이 쓰지 않는다. 생성기는 계획에 없는 세션 파일을 지우고 정답·매니페스트를 새로 쓰므로, 커밋된 세트에 그대로 쓰면 공식 세트가 바뀌기 때문이다. 다시 만든 `data/eval-dev-rebuild`도 `official: true`이므로, 한 번 더 만들려면 그 디렉터리를 지우고 돌린다. 코퍼스 없이 전 경로만 확인하려면 `make smoke`를 쓴다. 합성 음성과 `fallback_ism` 엔진을 쓰므로 산출물에 `official: false`가 기록되고 점수는 의미가 없다.

### 화자 구분 시스템 학습 (본 평가 준비)

비교 사다리의 네 시스템을 같은 학습·검증 코드로 돌린다. 모델 구조는 `models/`(시스템마다 파일 하나), 학습 방법은 `training/`, 시스템별 설정은 `configs/<이름>.json`에 있다. Colab에서는 `notebooks/train_template.ipynb` 하나가 이 코드를 불러 실행하며, 시스템을 바꿀 때는 노트북을 복사하지 않고 맨 위 `CONFIG`만 바꾼다.

| 설정 | 시스템 | 학습 |
|---|---|---|
| `voice_only` | 기준 채널(가장 낮은 번호의 활성 채널)의 1초 청크마다 동결 ECAPA 임베딩, 온라인 코사인 슬롯 매칭 | 없음 (임계값만 검증 세트에서 정한다) |
| `space_only` | 청크마다 Baseline 2와 같은 함수(`srp_phat_response`, `response_to_distribution`)로 만든 36구간 방위 분포, 같은 슬롯 매칭 | 없음 (임계값) |
| `naive_fusion` | 두 유사도를 마이크 수별 고정 가중치로 더한다 | 없음 (가중치와 임계값) |
| `proposed_carry`, `proposed_reset` | TAC 공간 경로 + 동결 ECAPA + 2층 인과 GRU, 시점마다 두 단서의 비중을 게이트로 정한다 | 학습 |
| `smoke` | 가짜 음성·간이 RIR·대체 인코더로 학습 경로 전체를 점검한다 | 학습 (숫자는 의미 없다) |
| `smoke_real` | 실제 ECAPA·`pyroomacoustics`·실제 크기 모델로 30걸음 (dev-clean만). Colab GPU에서 처음 한 번 돌리는 점검용 | 학습 (숫자는 의미 없다) |

- **공통 백엔드**: 모든 시스템은 시점마다 단위 벡터 하나를 내고, 같은 온라인 슬롯 매칭(`models/slots.py`)이 화자 슬롯을 붙인다. 그래서 사다리의 차이는 임베딩에 무엇을 넣었는지에서만 나온다. 임계값(단순 결합은 가중치도)은 시스템마다, 마이크 수마다 검증 세트에서 정한다.
- **제안 모델의 학습 방식**: 공간 경로에는 채널 사이 상대 위상·레벨 차와 마이크 좌표만 들어가고, 마이크가 1개면 정보가 없다. 그런데 위치를 고정해도 이런 특징만으로 화자를 꽤 가를 수 있었다(채널 간 특징으로 화자를 맞히는 선형 분류가 중앙값 약 0.88). 그래서 공간 경로는 화자 손실이 아니라 **음원 위치를 가르는 손실**로만 학습한다. 학습 세션의 일부에서는 서로 다른 화자가 같은 위치에서 말하게 해, 같은 위치면 목소리가 달라도 같게 보도록 만든다. 화자 손실은 게이트만 학습한다. 게이트는 목소리를 입력으로 받지 않고(공간 상태, 마이크 수, 발화 비율만 받는다) 마이크 1개에서는 구조상 0이다. 그래서 마이크 1개의 제안 모델은 목소리 전용과 같다(`tests/test_training.py`). 두 손실에는 같은/다른 화자 판정으로 유사도의 절대 척도를 맞추는 항을 더했다. 슬롯을 열지 말지는 이 척도 위에서 정해진다. 그 척도가 가리키는 임계값은 `theta_learned`로 기록만 하고, 실제 슬롯 임계값은 다른 시스템처럼 검증 세트에서 정한다.
- **데이터**: 학습 화자는 `LibriSpeech` `train-clean-100`(251명)이다. 검증 화자는 `dev-clean`에서 사전 측정 평가셋 화자 6명(`simulate.assign_splits`, 시드 `20261029`)을 뺀 34명이다. 본 평가셋 화자는 `test-clean` 40명이고, 학습·검증 코퍼스에 이들이 섞여 있으면 학습 코드가 시작을 거부한다(`training/corpus.py`). 그래서 평가에 쓰는 목소리는 학습과 임계값 조정 어디에도 쓰이지 않는다. 방 음향은 `pyroomacoustics`로 방 800개(방마다 마이크 4개, 음원 위치 4곳, 위치마다 0.1\~0.3 m 옮긴 사본 2개)를 한 번 계산해 둔다. 학습 세션에서는 이 방들에 매번 새 음성을 섞되 아래 조건을 섞는다. 배열 모양, 방 크기, 이득 규칙은 `simulate.py`의 것을 그대로 쓴다.
  - 턴마다 위치 사본 중 하나를 쓴다. 화자가 완전히 정지해 있으면 그 자리의 반사 패턴만으로 화자가 갈려(거울상 위치도 갈렸다) 모델이 방향이 아니라 시뮬레이터를 배운다.
  - 확산 잡음(SNR 0\~20 dB)을 넣는다.
  - 화자 순서는 무작위다(같은 화자가 연달아 말할 수 있다). 엄격히 번갈아 말하면 "쉰 뒤에는 다른 사람"이라는 지름길이 생긴다.
  - 절반은 평가셋 꼴(2명, 가까운 쪽과 먼 쪽, 2\~13초 턴 2\~4개), 절반은 짧은 턴이 많은 꼴(2\~4명)이다.
- **검증 세트**: 본 평가셋과 같은 조건(2명, 가까운 쪽과 먼 쪽, 긴 턴, 턴마다 위치 사본, 확산 잡음 SNR 5·10·20 dB)을 검증 화자로 만든 200세션이다. 임계값·가중치와 최고 체크포인트(마이크 2개 이상 케이스의 평균)를 이 세트로 정한다. 정지·무잡음 조건을 주 조건으로 삼지 않은 이유는 그 조건에서 학습 없는 단순 결합이 이미 causal 상한에 1점 안쪽으로 붙어(상한 0.949, 단순 결합 0.938\~0.948, 목소리 전용 0.87, 200세션 실측) 학습의 이득을 잴 여지가 없기 때문이다. 제안 모델의 최종 보고에는 같은 세션을 바꾼 조건을 진단으로 덧붙인다. 진단은 모두 같은 세션끼리 짝지어 본다(`summary.json`의 `diagnostics`).
  - 정지·무잡음 조건 (본 평가셋의 정지 조건과 같은 꼴, 임계값은 이 조건에서 다시 정한다)
  - 모든 화자를 한 위치에 겹친 조건: 공간 경로만으로는 우연 수준이어야 하고, 전체 모델이 목소리 전용 아래로 떨어지면 안 된다.
  - 목소리 전용 대비 이득
  - 학습된 공간 경로만 쓴 성능
  - 게이트를 마이크 수별 고정 가중치로 바꾼 성능
- **라벨과 상한**: 학습 손실은 각 시점까지 실제로 들린 화자(`[t_k − 0.25, t_k)`)를 쓰고, 검증 정확도는 공식 정답 규칙(`[t_k, t_k + 0.25)`)으로 잰다. causal 시스템은 화자가 바뀌는 순간의 한 시점을 원리상 맞힐 수 없다. 들은 화자를 이어 부르는 causal 오라클은 평가셋 꼴 검증 세션에서 약 0.95다(`summary.json`의 `causal_ceiling_by_N`). 이 값은 라벨 규칙의 활성 기준으로 "들은 화자"를 정한 근사치라 엄밀한 상한은 아니다. 활성 기준보다 조용한 발화 시작도 실제 오디오에는 들리므로, 본 평가셋 정지 조건에서는 학습 없는 단순 결합이 최대 0.964로 이 값을 넘는다. 학습에 섞는 짧은 턴 세션에서는 2인 0.92, 3인 0.90, 4인 0.86이다(각 50\~300세션 실측). 검증은 기본이 2인이다. 3인 이상을 보려면 `data.val_n_speakers`를 올리고 `data.val_eval_layout_prob`를 1보다 낮춰야 한다.
- **공식 채점기와의 관계**: 검증 화자 정확도는 슬롯과 화자를 가장 잘 맞게 짝지어 잰다. 2인 검증 케이스는 `evaluate.py --track 2`에도 넣어 채점한다(`official_n2`). 이때 임계값은 현재 채점 계약대로 슬롯을 2개로 묶은 조건에서 마이크 수별로 다시 정한다. 전사가 없는 예측이라 공식 채점기도 같은 짝을 고른다. 그래서 같은 조건의 내부 값(`spk_acc_n2_cap2`, 케이스 평균)과 공식 값이 일치한다(`tests/test_training.py`). 3인 이상은 현재 채점 계약(세션당 2인)으로 채점할 수 없어 내부 지표로만 본다.

```bash
make train-smoke   # 코퍼스·체크포인트 없이 학습 -> 검증 -> 요약까지 CPU로 1~2분 (runs/train-smoke)
make train-test    # 인과성, 채널 순서 불변, 마이크 1개 = 목소리 전용, 렌더러 일치, 체크포인트·뱅크 복구 테스트
make train TRAIN_CONFIG=configs/voice_only.json TRAIN_DATA=<LibriSpeech/ 를 담은 디렉터리>
```

실행 디렉터리(`--run-dir`)에 남는 것:

- `config.json`: 설정과 지문
- `ckpt/`: 재개용 체크포인트. 걸음 수나 10분 간격으로 저장하고 최근 3개를 남긴다. `LATEST.json`에 크기와 SHA-256을 함께 적는다.
- `best.pt`: 검증 최고 가중치
- `metrics.jsonl`, `summary.json`
- `val_cases.jsonl`: 케이스별 기록. 가설 H1\~H4를 같은 검증 세트에서 나눠 볼 수 있게 다음을 함께 적는다.
  - 마이크 수, 화자 수, 두 화자의 성별
  - 잔향 없는 음성으로 잰 두 화자의 목소리 유사도
  - 방위 차, 마이크 쌍 기준 도달 시간차
  - 턴 수와 교대 횟수
  - 교대 직후 1초와 턴 안쪽의 정확도

같은 명령을 같은 실행 디렉터리로 다시 돌리면 크기와 해시가 맞는 가장 최근 체크포인트에서 이어 간다. 업로드가 덜 된 파일이면 그 이전 체크포인트로 돌아간다. 설정이 바뀌었으면 이어 가지 않는다. 코드 커밋이 바뀌었을 때도 `--allow-code-change` 없이는 이어 가지 않으며, 쓴 커밋은 모두 기록된다.

검증 프로토콜은 현재 `v3`이다. 학습·검증의 확산 잡음은 배포 평가셋과 같은 Hann overlap-add 정의를 사용하며, 기존 평가셋 파일은 그대로 쓴다. 이전 프로토콜의 요약과 점수는 재검증·재채점한다. 호환되는 이전 `best.pt`와 최신 체크포인트는 새 기준으로 각각 검증해 더 좋은 가중치를 보존하고, 학습 재개용 모델·옵티마이저·난수 상태는 유지한다. 평가 결과를 재사용할 때는 ASR 백엔드도 같아야 한다.

**Colab에서**: 노트북 맨 위 `CONFIG`만 바꿔 실행한다. 노트북은 다음을 처리한다.

- 학습 환경은 Colab의 `torch`를 사용하고 `speechbrain`을 설치한다. 최신 ASR은 별도 venv에서 실행한다.
- LibriSpeech는 Drive에 캐시하고 매번 로컬 디스크에 풀며, OpenSLR md5와 대조한다.
- 방 음향 뱅크는 Drive에 50개 방 단위로 미러링하므로 끊겨도 이어서 만든다.
- 체크포인트와 요약은 Drive에 남긴다.

### 최신 ASR와 단일채널 화자 구분 비교

`notebooks/train_template.ipynb`의 기본 전사는 **Qwen3-ASR-1.7B**(`ASR="qwen3-1.7b"`)다.
기존 baseline 3종과 proposed 3종 모두 같은 고정 전사 모델을 쓴다. Qwen 자체는 화자를 구분하지 않으며,
각 시스템이 예측한 화자 슬롯의 연속 구간을 전사한다. `ASR="wav2vec2"`로 이전 전사를 재현할 수 있다.
ASR 교체는 화자 모델을 재학습하지 않고 임베딩·체크포인트를 재사용해 전사와 점수만 갱신한다.
기존에 기록한 wav2vec2 수치와 새 Qwen 수치는 백엔드를 명시해 구분한다.

최신 단일채널 비교는 `notebooks/mono_baselines.ipynb`에서 다음 세 가지를 실행한다.

| 비교군 | 화자 구분 | 전사 | 실제 입력 |
|---|---|---|---|
| Nemotron + Qwen | [NVIDIA Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), 기존 시스템과 같은 slot-run 규칙 | 1ch |
| VibeVoice | [Microsoft VibeVoice-ASR 9B](https://huggingface.co/microsoft/VibeVoice-ASR) | 모델 자체 전사 | 1ch |
| MOSS | [OpenMOSS MOSS-Transcribe-Diarize 0.9B](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) | 모델 자체 전사 | 1ch |

먼저 `LIMIT=2`로 GPU 추론·재개·채점을 확인한 뒤 `LIMIT=0`으로 전체 평가한다.
VibeVoice 전용 기존 진입점 `notebooks/track_b_vibevoice.ipynb`도 같은 실행기를 사용한다.
전체 비교군 실행에는 A100을 권장한다. VibeVoice는 가중치만 약 18GB이므로 T4용 설정이 아니다.
공통 Qwen은 CUDA GPU 종류가 바뀌어도 전사 조건을 유지하도록 FP16을 사용한다. VibeVoice/MOSS는
Ampere 이상에서 BF16을 선택한다. 실제 VRAM·속도와 모델 설치는 Colab에서 확인해야 한다.

**비교 조건**: 같은 세션·같은 기준 마이크(활성 채널 중 최소 번호)를 사용한다. mono 비교군은
20초 전체를 본 `full_audio` 결과이고, proposed의 화자 결정은 `causal`이다. 관측 범위를 표시해 별도 행으로 보고한다.
Nemotron은 streaming도 지원하지만, 이번 기본 프리셋은 offline이다. `by_comparison_M`은 짝지은 proposed의
마이크 수이며, mono 모델이 입력받은 마이크 수는 언제나 1이다. proposed의 1ch 결과도 함께 보고한다.
모델이 화자 2명을 초과해 출력하면 발화 시간 상위 두 화자를 유지하고 나머지는 최장 화자에 합치는
`top2-duration-fold-v1` 후처리를 적용한다. 정답·정답 화자 ID·정답 전사는 추론에 사용하지 않는다.
최신 모델 대비 우위는 사전에 가정하지 않고 SpkAcc·cpWER 실측으로 판단한다.

명령행에서는 학습 인터프리터를 바꾸지 않고 모델별 환경을 만든다. 아래는 Colab 경로 기준이다.

```bash
python3 setup_asr_env.py --profile qwen --root /content/vcsasr-envs
python3 eval_t2.py --run-dir /content/drive/MyDrive/vcsasr/runs/proposed_carry \
  --out /content/drive/MyDrive/vcsasr/eval/proposed_carry --device cuda \
  --asr qwen3-1.7b --asr-python /content/vcsasr-envs/qwen/bin/python \
  --asr-cache /content/drive/MyDrive/vcsasr/asr_cache

python3 setup_asr_env.py --profile nemotron --root /content/vcsasr-envs
python3 mono_baselines.py --backend nemotron --python /content/vcsasr-envs/nemotron/bin/python \
  --qwen-python /content/vcsasr-envs/qwen/bin/python --limit 2 \
  --out /content/drive/MyDrive/vcsasr/mono_baselines/nemotron \
  --asr-cache /content/drive/MyDrive/vcsasr/asr_cache

python3 compare_asr.py --eval-root /content/drive/MyDrive/vcsasr/eval \
  --mono-root /content/drive/MyDrive/vcsasr/mono_baselines \
  --out /content/drive/MyDrive/vcsasr/asr_comparison.json
```

VibeVoice/MOSS는 각각 같은 이름의 환경을 만든 뒤 `--backend`와 `--python`을 바꾼다.
`--qwen-python`은 Nemotron에만 필요하다. 모델·공식 소스 revision은 `modern_asr.py`에 고정하고,
출력에 오디오 SHA-256, 디코딩 설정, 실제 패키지 버전·정밀도·입력 채널·관측 범위를 기록한다.
이들이 바뀌면 다른 출력/캐시 경로를 사용한다(노트북의 `RUN_TAG`, `ASR_CACHE_NAME`). 같은 파일에는 한 실행만 기록해야 한다.
중단된 마지막 JSONL 조각은 복구하지만, 정상 줄의 손상과 파싱 실패는 중단시킨다.
VibeVoice/MOSS의 파싱 오류나 토큰 상한에 도달한 출력을 완료로 캐시하지 않는다. 라이브러리 목록은 각 venv의
`resolved-requirements.txt`와 mono 노트북의 Drive `mono_baselines/runtimes/`에 남는다.
노트북의 가중치 캐시는 Drive `models/asr/`에 보존한다. CLI에서는 `VCSASR_MODEL_CACHE`로 같은 경로를 지정할 수 있다.
비교 표의 case hash와 표본 수가 같아야 같은 평가 집단이다. 시험 실행과 전체 실행을 직접 비교하지 않는다.

로컬에서는 단위·종단 테스트와 노트북 문법, 네 모델 환경의 실제 설치·import를 확인했다.
Nemotron은 고정 체크포인트로 평가셋의 20초 오디오 한 개를 CPU에서 추론했다.
Qwen/VibeVoice/MOSS의 실제 GPU 전사와 전체 평가 성능 비교는 Colab에서 확인해야 한다.

`pyroomacoustics` 0.8.5는 Linux 설치 파일이 없다. 그래서 뱅크가 Drive에 없을 때만 Git 태그에서 빌드한다(이 맥에서 태그 빌드는 휠과 RIR이 비트 단위로 같았다). 학습 세션의 방 음향 적용(컨볼루션, 레벨, 클리핑 방지, 16비트 양자화)은 GPU에서 배치째 한다. CPU 작업자는 음성 디코딩과 발화 배치만 맡는다. 그래서 작업자 쪽 일이 세션당 0.25\~0.5초에서 10\~19 ms로 줄었다(이 맥, 화자 2\~4명). 같은 float32 입력이면 CPU 기준 구현(`training/scenes.render_numpy`)과 16비트 단위로 2 이하만 다르다(`tests/test_training.py`). 학습 입력은 float16으로 넘기므로 최대 15 단위(RMS 1 미만)까지 다르다. 처음에는 `smoke_real`로 설치, 혼합 정밀도, 메모리, 데이터 준비 시간 대 GPU 계산 시간을 확인한다.

`Makefile`의 주요 변수(`TRACK`, `SCOPE`, `BASELINE`, `N_SESSIONS`, `SEED`, `DATA_DIR`, `GEN_DIR`, `LIBRISPEECH_ROOT`, `GT`, `PRED`, `SCORES`, `SIM_SESSIONS`, `SMOKE_DIR`, `SMOKE_SESSIONS`, `TRAIN_CONFIG`, `TRAIN_DATA`, `TRAIN_RUN`, `TRAIN_ARGS`)는 명령행에서 덮어쓸 수 있으며, 주요 타깃 설명은 `make help`로 확인할 수 있다.

> **CLI 플래그 및 인터페이스 정합성 메모** (`--help` 기준 검증 완료):
>
> - `make baseline`: 베이스라인 1·2 모두 `--eval-dir / --out / --scope / --track` 인자를 받는다. `BASELINE=2`의 모듈 경로는 `baselines.dsp_srp_phat`이다.
> - `make score`: `evaluate.py`의 인자는 `--pred / --gt / --track / --out / --indent`다. 정답 파일 인자는 `--gt`이며 별도의 `--scope` 플래그는 두지 않는다. 평가기는 제출 레코드 내부의 `observation_mode` 필드(생략 시 `causal`)를 읽어 `spec.OFFICIAL_SCOPES` 기준으로 공식·비공식 집계를 직접 분리한다.
> - `make eval-dev`: 평가셋 생성 엔트리포인트는 `make_eval_dev.py`(`--out / --n-sessions / --seed / --librispeech-root / --synthetic / --rir-engine / --verify-only / --overwrite-official`)이며, `simulate.py`는 음향 합성 및 자체 점검용 라이브러리다.
> - `MANIFEST`: 베이스라인 실행 시 `--manifest $(DATA_DIR)/meta.jsonl`을 명시적으로 전달한다. 디렉터리 내 요약 파일인 `manifest.json`과 케이스별 입력 목록인 `meta.jsonl`을 혼동하지 않도록 분리하기 위함이다.
> - `SCOPE` 기본값은 문자열 하드코딩 대신 `spec.OFFICIAL_SCOPES[0]`에서 읽어 온다.

---

## 제출 파일 스키마

제출 파일(`predictions.jsonl`)은 한 줄에 JSON 객체 하나씩 기록하는 **JSONL** 형식이며 두 종류의 레코드로 구성된다. 화자 구분 주 지표는 `timepoint` 레코드의 `pred_speaker`(`Spk Acc`)와 `final_transcripts` 레코드의 화자별 전사(`cpWER`)로 계산하고, 방위·거리 필드는 공간 진단 지표에만 쓰인다. 아래 예시는 `spec.py` 검증을 통과하는 실제 형식(`azimuth_probs` 36개, `distance_probs` 8개, 각 합 1.0)으로 쓴 단일 화자 케이스 예시다.

### 레코드 1: `timepoint` — 격자 시점별 화자·공간 예측

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

위 예시 값은 채널 0·2가 x축 위에 마주 놓인 기준 배열(`spec.DEFAULT_ARRAY_GEOMETRY`, 균일 원형)을 가정한 설명용이며, 커밋된 eval-dev의 `sess_0001`은 배열 방향과 정답 방위가 이와 다르다. 마스크 `1010`은 채널 0과 채널 2만 사용하는 2채널(`M = 2`) 조건이다. 전후방 대칭성으로 인해 방위 분포가 구간 14(중심 −35°)와 구간 21(중심 +35°) 두 봉우리로 갈라지며, `spec.INTERVAL_RULE`에 따른 90% HPD 구간 집합은 `{12, 13, 14, 15, 16, 20, 21, 22}`(비연속 구간, 총 폭 80°, 누적 확률 0.9024)가 된다. 정답 방위각이 −37.4°(구간 14)라면 커버리지는 1, NLL은 1.6274로 계산된다.

### 레코드 2: `final_transcripts` — 케이스별 최종 화자 전사

케이스당 1개(세션당 6개)를 기록한다. 세션당 2인 케이스(`--track 2`)라면 `{"s0": "...", "s1": "..."}`처럼 화자 슬롯마다 전사를 적는다.

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
| `record_type` | `string` (`"timepoint"`) | 권장 | - | - | 생략 시 `eval_time_sec`이 있으면 `timepoint`로 추론한다(`final_transcripts`가 함께 있어도 `timepoint`로 읽고 전사는 무시한다). 둘 다 없으면 거부(`R-PARSE`). `"timestep"` 하위 호환 별칭 읽기 지원 |
| `case_id` | `string` (`<session_id>__m<mask_bits>`) | 필수 | - | 전 지표 1차 키 | 스키마 오류로 파일 거부 |
| `eval_time_sec` | `float` (`k * spec.DELTA_T_SEC`, 0.25 배수) | 필수 | 초 (`session_start` 기준) | 화자·공간 지표 2차 키 | 스키마 오류로 파일 거부 |
| `pred_speaker` | `string` (세션 내 익명 화자 슬롯; `speaker_id`, `speaker` 별칭 수용) | 선택 | - | `speaker_attribution_accuracy` (`Spk Acc`, 화자 구분 주 지표) | `spec.PENALTY_SPEAKER_CORRECT = False` (오답 처리) |
| `azimuth_probs` | `array[36]` of `float` | 선택 | 확률 (합 `1 ± 1e-4`) | `azimuth_nll`, `azimuth_coverage_90`, `azimuth_interval_width_deg` | 레코드 결측 시 `PENALTY_NLL`(27.631) 부과; 점 전용 제출 시 분포 지표 `N/A` |
| `distance_probs` | `array[8]` of `float` | 선택 | 확률 (합 `1 ± 1e-4`) | `distance_nll`, `distance_coverage_90`, `distance_interval_width_m` | 레코드 결측 시 `PENALTY_NLL` 부과; 점 전용 제출 시 분포 지표 `N/A` |
| `pred_azimuth_deg` | `float` (`[-180, 180)`) | 선택 | 도 (°) | `azimuth_mae_deg`, `azimuth_acc_10deg`, `azimuth_acc_20deg`, `pos_error_2d_m` | 생략 시 `azimuth_probs` 최빈(MAP) 구간의 `AZIMUTH_BIN_CENTERS_DEG` 사용 (`R-PT`) |
| `pred_distance_m` | `float` (`[0.5, 5.0]`) | 선택 | m | `distance_mae_m`, `pos_error_2d_m` | 생략 시 `distance_probs` 최빈(MAP) 구간의 기하평균 중심 `DISTANCE_BIN_CENTERS_M` 사용 (`R-PT`) |

> 2차원 직교좌표(`pred_x_m`, `pred_y_m`)도 선택 필드로 함께 또는 단독 제출할 수 있으며, 극좌표와 직교좌표를 동시에 기재한 경우 변환 정합성을 검사한다(`R-POS-02`).

`final_transcripts` 레코드:

| 필드명 | 타입 | 필수 여부 | 단위 | 연관 지표 | 생략·누락 시 처리 |
|---|---|---|---|---|---|
| `record_type` | `string` (`"final_transcripts"`) | 권장 | - | - | 생략 시 `eval_time_sec` 없이 `final_transcripts` 키가 있으면 이 타입으로 추론 |
| `case_id` | `string` | 필수 | - | `wer`, `cpwer` 키 | 스키마 오류로 파일 거부 |
| `observation_mode` | `string` (`causal`, `windowed`, `full_audio`) | 선택 | - | 관측 범위별 분리 집계 | 생략 시 `causal`로 간주. 허용 값 밖이거나 `windowed`인데 `lookahead_sec`이 없으면 파일 거부(`R-PARSE`) |
| `final_transcripts` | `object` (`{speaker_id: string}`, 슬롯 수 ≤ 화자 수 상한) | 필수 | - | `cpwer` (화자 구분 주 지표), `wer` (`micro_global`) | 케이스 누락 시 참조 단어 전체를 삭제 오류로 계산 (`count_all_deletions`) |
| `rtf` | `float` (추론 시간 / 오디오 길이) | 권장 | 배수 | `rtf` | 모든 케이스에서 미기재하면 `rtf`를 `N/A`로 표기하고, 일부 케이스만 미기재하면 그 케이스는 `rtf` 평균에서 빠진다 |

화자 무관 `wer`는 화자 슬롯별 전사를 슬롯 이름 오름차순으로 이어 붙여 계산한다(`MULTI_SPEAKER_WER_CONCAT_ORDER`). 모든 정식 필드명과 별칭 해석 순서는 `spec.py`(`F_*` 상수, `FIELD_ALIASES`, `REQUIRED_FIELDS`, `PROB_FIELD_LENGTHS`)에 단일 정의되어 있다(세부 결정 근거는 `SPEC_DECISIONS.md` 13절 참조).

---

## 채점 규칙 요약

`spec.py`는 규칙을 두 종류로 나눈다. 거부 규칙(`spec.REJECTION_RULES`)에 걸리면 `evaluate.py`가 제출 파일 전체를 거부하고 종료 코드 2로 끝나며, 오류 메시지는 규칙 ID로 시작한다(예: `R-PROB-02: predictions.jsonl:1 azimuth_probs: probabilities sum to 1.088, tolerance is 0.0001`). 채점 규칙(`spec.SCORING_RULES`)은 점수를 계산하는 방식을 정한다.

거부 규칙:

| 규칙 ID | 거부 조건 |
|---|---|
| **R-PARSE** | 레코드를 읽을 수 없거나 타입이 틀렸거나 필수 필드가 없다 |
| **R-DUP-01** | 같은 `case_id`의 최종 전사 레코드가 둘 이상이다 |
| **R-DUP-02** | 같은 `(case_id, eval_time_sec)`의 시점 레코드가 둘 이상이다 |
| **R-PROB-01** | 확률 벡터 길이가 36(방위) / 8(거리)이 아니다 |
| **R-PROB-02** | 확률 합이 `1 ± 1e-4`(`PROB_SUM_TOLERANCE`)를 벗어난다. 평가기는 재정규화하지 않는다 |
| **R-PROB-03** | 확률 값이 `[0, 1]` 밖이다 |
| **R-SPK-01** | 케이스의 화자 슬롯 수가 화자 수 상한(`MAX_SPEAKERS_PER_TRACK`: 단일 화자 1, 2인 2)이나 순열 매칭 상한(`MAX_PERMUTATION_SLOTS`)을 넘는다 |
| **R-POS-01** | 직교좌표 위치에 `x`, `y` 중 하나만 있다 |
| **R-POS-02** | 극좌표와 직교좌표로 함께 적은 위치가 서로 맞지 않는다 |

채점 규칙:

| 규칙 ID | 요약 | 관련 `spec.py` 상수 |
|---|---|---|
| **R-DUP** | 중복 레코드가 있으면 임의의 행을 선택하지 않고 제출 파일 전체를 거부한다(`R-DUP-01`·`R-DUP-02`). | `DELTA_T_SEC`, `DUPLICATE_RECORD_POLICY` |
| **R-MISS** | 활성 평가 시점 결측 시 레코드를 제외하지 않고 최악 페널티(방위 180.0°, 거리 5.0 m, 2D 오차 10.0 m, 화자 오답, NLL 27.631 nats)를 부과한다. 전사 누락 시에는 참조 단어를 전량 삭제 오류로 계산해 전략적 기권 이득을 차단한다. | `PENALTY_AZIMUTH_DEG`, `PENALTY_DISTANCE_M`, `PENALTY_POS_ERROR_2D_M`, `PENALTY_NLL`, `MISSING_TRANSCRIPT_POLICY` |
| **R-PERM** | 예측·참조 화자 슬롯을 `max(len(pred), len(ref))` 크기로 빈 문자열 패딩한 뒤 1:1 최적 순열을 지표마다 따로 찾는다. `cpwer`은 `cpWER 최소 → 시점별 화자 일치 수 최대 → 참조 슬롯 인덱스 사전순`, `Spk Acc`는 `시점별 화자 일치 수 최대 → 참조 슬롯 인덱스 사전순`이라 전사가 화자 귀속 정확도를 바꾸지 못한다(spec 0.3.0). 단일 화자(`--track 1`)의 `cpwer`는 규칙상 `N/A`다. | `SPEAKER_PAD_TOKEN`, `PERMUTATION_TIE_BREAK`, `MAX_SPEAKERS_PER_TRACK`, `TRACK_UNDEFINED_METRICS` |
| **R-NLL** | 제출 확률을 평활화하거나 재정규화하지 않는다. 수치 안정성을 위한 `1e-12` 하한 클리핑은 정답 구간 확률에만 적용한다. | `K_AZIMUTH`, `K_DISTANCE`, `NORMALISE_SUBMITTED_PROBS`, `NLL_EPSILON` |
| **R-PT** | 명시적 점 예측이 있으면 우선 사용하고, 생략된 경우에만 분포의 MAP 구간 대표점(방위 산술 중심, 거리 기하 중심)으로 점 지표를 계산한다. | `AZIMUTH_BIN_CENTERS_DEG`, `DISTANCE_BIN_CENTERS_M` |
| **R-TXT** | `wer`는 `final_transcripts`만 읽으며, 참조와 가설 텍스트 모두에 `NFKC → 소문자화(casefold) → 구두점 처리 → 숫자 영단어 변환(0~999) → 공백 정리`를 동일 순서로 적용한다. | `TEXT_NORMALISATION`, `WER_AGGREGATION` |

집계 방식 구분:

- **화자 귀속·공간 지표 (`macro_per_case`)**: `Spk Acc`와 공간 지표는 각 케이스 내 활성 발화 시점(`T_eval`)에 대해 균등 평균을 낸 뒤, 전체 케이스에 동일 가중치를 부여해 평균한다. 발화 길이가 긴 특정 세션이 전체 점수를 지배하지 않도록 하기 위함이다.
- **전사 오류율 (`micro_global`)**: `wer`와 `cpwer`는 표준 ASR 관례에 따라 전체 케이스의 편집거리 총합을 전체 참조 단어 수로 나눈다.
- **불확실성 구간 (`hpd_greedy`)**: 확률 내림차순으로 구간을 선택해 누적 확률이 0.90에 도달하는 최소 집합을 구성한다. 비연속 구간을 허용하므로 `M = 2` 양봉 분포의 두 후보 방향이 중간 빈 구간 없이 정확히 반영된다.

---

## eval-dev 배포 규격 (사전 측정 평가셋)

### 오디오 및 라벨 용량 산정

4채널 · 16 kHz · 16-bit PCM(`pcm_s16le`) 기준 초당 데이터양은 다음과 같다.

```text
128,000 B/s = 4 ch × 16,000 Hz × 2 B  (= 128 kB/s, 125 KiB/s)
```

세션 길이는 `spec.SESSION_LENGTH_SEC = 20.0`초로 고정되므로 세션 수에 따른 용량은 아래와 같다.

| 항목 | 계산식 | 용량 |
|---|---|---|
| 세션 1개 오디오 (4ch WAV) | `128,000 × 20` + WAV 헤더 44 B | **2.56 MB** (2.441 MiB) |
| 세션 1개 정답 라벨 (6케이스, 활성 시점 `T_eval`만 기록) | `ground_truth.jsonl` 실측 평균 | 약 **38.7 kB** |
| 세션 100개 | 오디오 기준 | 256.0 MB |
| **공식 배포본 (144세션)** | 오디오 368.6 MB + 라벨·메타 6.2 MB | **374.9 MB** (357.5 MiB) |

오디오 파일은 마스크별로 중복 저장하지 않고, 4채널 세션을 1회 저장한 뒤 로더가 `spec.mask_to_channels`로 활성 채널만 추출한다(마스크 6종의 활성 채널 합은 13개이므로 개별 저장 시 용량이 3.25배 증가함). 추론 결과인 `predictions.jsonl`은 확률 벡터를 포함해 864케이스 기준 Baseline 1이 74.3 MB, Baseline 2가 65.7 MB이므로 `.gitignore`로 버전 관리에서 제외한다.

### 저장소 용량 기준

| 구분 | 기준값 | 설정 근거 |
|---|---|---|
| **평가셋 한 벌 용량** | **≤ 550 MB** | 세 벌을 합쳐도 GitHub가 강하게 권고하는 저장소 규모(5 GB)와 한 번에 올릴 수 있는 양(2 GB) 안에 둔다 |
| **단일 파일 크기** | **≤ 90 MB** | GitHub 파일당 업로드 제한(100 MiB) 미만의 안전 한계 |
| **`eval-dev` 커밋 규모** | **374.9 MB** (357.5 MiB) | 144세션 개별 WAV(2.44 MiB) + JSONL 라벨 및 해시 매니페스트 |
| **`eval-t2`, `eval-t2-static` 커밋 규모** | 각 **523.2 MB** (499.0 MiB) | 200세션 개별 WAV(2.44 MiB) + JSONL 라벨(가장 큰 파일이 정답 9.84 MiB) 및 해시 매니페스트 |
| **평가셋 합계** | **1,421.3 MB** (1,355.5 MiB) | GitHub가 이상적이라고 하는 1 GB는 넘고, 강하게 권고하는 5 GB 안이다 |

### 커밋 포함 및 제외 대상

챌린지 제출 요건(개인 저장소 내 평가용 데이터셋 직접 포함)에 맞춰 `data/eval-dev/`, `data/eval-t2/`, `data/eval-t2-static/`의 평가 입력과 정답을 저장소에 함께 포함한다.

| 경로 | 설명 | Git 추적 여부 |
|---|---|---|
| `data/eval-dev/audio/` | 144세션 4채널 WAV (총 351.6 MiB, 파일당 2.44 MiB) | 포함 |
| `data/eval-dev/ground_truth.jsonl` | 공식 채점용 정답 라벨 (5.31 MiB) | 포함 |
| `data/eval-dev/meta.jsonl` | 추론 입력용 케이스별 메타데이터 (0.58 MiB) | 포함 |
| `data/eval-dev/manifest.json` | 파일별 SHA-256 해시 및 검증 통계 요약 (0.05 MiB) | 포함 |
| `data/eval-dev/ATTRIBUTION.md` | `LibriSpeech` 파생 오디오 출처 및 라이선스 고지 | 포함 |
| `data/eval-dev/rir/` | 합성 중간 산출물(15.0 MiB, 동일 시드로 재생성 가능) | 제외 |
| `data/eval-dev-rebuild/` | `make eval-dev`가 다시 만든 사본 | 제외 |
| `data/eval-t2/`, `data/eval-t2-static/` | 본 평가셋 주 조건·정지 조건. 각각 200세션 4채널 WAV(488.3 MiB)와 정답·메타·매니페스트·출처 고지 | 포함 |
| `data/eval-t2-rebuild(-static)/` | `make eval-t2`가 다시 만든 사본 | 제외 |
| `runs/` | 추론·채점 중간 결과물과 `make smoke` 산출물 | 제외 |

### Git LFS 대신 일반 Git 추적을 선택한 이유

합계 1.3 GiB 규모의 오디오 파일은 통상 Git LFS 대상이지만, 공개 평가용 저장소에서는 Git LFS가 오히려 다운로드 장애를 유발할 수 있다. Git LFS 다운로드 대역폭은 누가 내려받든 저장소 소유자의 한도에서 차감되고, GitHub Free 계정에 포함된 대역폭은 월 10 GiB다(GitHub Docs 기준). 저장소를 한 달에 30번 정도만 복제(clone)해도 이 한도가 소진되고, 결제 수단이 없으면 다음 달까지 계정의 Git LFS가 꺼져 이후 채점자나 재현자가 오디오를 내려받지 못하게 된다.

개별 WAV 파일 크기가 2.44 MiB로 GitHub 단일 파일 제한(100 MiB)보다 충분히 작고, 고정 시드로 만든 공식 세트(`official: true`)는 두 생성기 모두 덮어쓰기를 막으므로 일반 Git 객체로 직접 추적하여 복제 가용성을 보장했다. 대신 저장소 전체를 복제하면 평가셋만 약 1.4 GB를 내려받는다.

### 본 평가셋 만들기

`make eval-t2`는 `LibriSpeech` `test-clean`(<https://www.openslr.org/12>, 화자 40명)과 `pyroomacoustics` 0.8.5로 본 평가셋을 만든다. 같은 세션을 두 벌로 쓴다. `data/eval-t2-rebuild`가 주 조건, `data/eval-t2-rebuild-static`이 정지 조건이다. 이 맥에서 8개 프로세스로 약 3분 걸렸고, 조건마다 499.0 MiB다. 같은 기본값으로 만든 공식 세트가 `data/eval-t2`, `data/eval-t2-static`에 커밋돼 있고, 다시 만든 사본이 이것과 바이트 단위로 같았다(`make eval-t2-check`).

```bash
make eval-t2-verify   # 렌더링 없이 계획의 균형만 확인한다 (numpy 불필요, 아무것도 쓰지 않는다)
make eval-t2 TESTCLEAN_ROOT=<test-clean.tar.gz 를 푼 디렉터리>/LibriSpeech/test-clean
make eval-t2-check    # 다시 만든 두 벌을 data/eval-t2, data/eval-t2-static 과 바이트 단위로 비교한다
```

- **세션**: 200개(케이스 1200개). 세션마다 새 방이고, 화자는 가까운 쪽(0.5\~1.9 m)과 먼 쪽(1.9\~5.0 m) 두 명이다. 방위 구간, 거리 구간, 잔향 대역, SNR, 턴 수를 세션에 고르게 나눈다. 화자 40명은 10세션씩 나오고, 성별 조합은 남녀 102, 여여 49, 남남 49세션이다.
- **턴**: 2·3·4개가 67·67·66세션이다. 순서는 무작위라 같은 화자가 연달아 말할 수 있다. 화자 교대는 1·2·3번이 134·55·11세션이다.
- **음성과 전사**: 화자마다 발화 하나를 통째로 쓰고, 쉼 지점에서 턴 수만큼 나눈다. 그래서 참조 전사는 원문 전사 그대로이고, 세션 사이에 같은 발화를 다시 쓰지 않는다(발화 400개). 쉼은 90번째 백분위 프레임보다 30 dB 낮은 0.18초 이상 구간이다. 이 기준으로 자를 발화가 없을 때만 25 dB, 20 dB·0.12초 순으로 완화한다(화자 자리 400개 중 10개, `ground_truth.jsonl`의 `pause_rule_db_s`).
- **주 조건**: 화자마다 위치 사본 3개(첫 위치, 그것에서 0.1\~0.3 m 옮긴 사본 2개)를 두고 턴마다 하나를 쓴다. 다시 말하는 화자는 직전과 다른 사본에서 말한다(옮긴 거리 0.02\~0.58 m). 확산 잡음(구면 등방, 핑크)을 세션 전체 잔향 음성 전력 기준 SNR 5·10·20 dB로 더한다. 학습·검증의 잡음과 같은 정의다.
- **정지 조건**: 첫 위치만 쓰고 잡음이 없다. 음성, 방, 이득, 라벨 시점은 주 조건과 같다.
- **라벨**: 공식 규칙(`[t_k, t_k + 0.25)`)으로 붙이고, `spk_0`은 먼저 말한 화자다. 시점마다 위치는 그 시점 턴의 사본 위치다.

두 벌 모두 다음을 확인했다.

- 정답을 예측으로 넣으면 공식 채점기가 `Spk Acc` 1.0, `cpWER` 0을 낸다.
- 참조 전사가 원문 전사와 모두 같다.
- 라벨 시점 109,548개의 화자와 위치가 턴·사본과 모두 맞는다.
- 클리핑이 없다.
- 프로세스 하나로 다시 렌더링한 세션 4개도 바이트 단위로 같았다.

들은 화자를 이어 부르는 causal 오라클은 0.948(전체 시점)이다. 다만 이 값은 근사치다(위 "라벨과 상한").

학습 없는 비교 시스템을 이 세트에 돌린 값은 다음과 같다. 채점 계약대로 슬롯은 2개다. 임계값과 가중치를 이 세트에서 마이크 수별로 골랐으므로 낙관적인 값이다.

| 조건 | 목소리 전용 | 공간 전용 (M = 2 / 3 / 4) | 단순 결합 (M = 2 / 3 / 4) |
|---|---|---|---|
| 주 조건 | 0.833 | 0.803 / 0.853 / 0.878 | 0.833 / 0.860 / 0.878 |
| 정지 조건 | 0.891\~0.894 | 0.935 / 0.951 / 0.963 | 0.935 / 0.954 / 0.964 |

정지 조건에서는 학습 없는 공간 단서만으로 마이크 3개부터 오라클 값을 넘고, 2개에서도 1.3점 차이라 학습한 결합이 이길 여지가 없다. 주 조건에서는 단순 결합이 오라클보다 7\~12점 낮다. 그래서 주 조건을 본 평가로 쓰고, 정지 조건은 같은 세션의 이상적인 경우로 함께 보고한다. 마이크 2개 조건에서 두 화자의 도달 시간차가 1샘플 미만인 세션은 마스크 `1010`에서 14%, `1100`에서 22.5%다.

생성기(`make_eval_t2.py`)는 `official: true`인 디렉터리에는 `--overwrite-official` 없이 쓰지 않는다. 그래서 한 번 더 만들려면 `data/eval-t2-rebuild(-static)`을 지우고 돌린다. `official: true`는 `pyroomacoustics` 0.8.5, `test-clean` 화자 40명, 기본 세션 수·시드가 모두 맞을 때만 기록된다. 학습 코드는 학습·검증 코퍼스에 `test-clean` 화자가 섞여 있으면 시작을 거부한다(`training/corpus.py`의 `TEST_CLEAN_SPEAKERS`).

---

## 라이선스 및 출처 표기

### LibriSpeech 파생 오디오 고지

원천 음성 코퍼스인 `LibriSpeech`(`dev-clean`, `test-clean`)는 **CC BY 4.0** 라이선스로 배포된다. 실내 임펄스 응답을 합성한 다채널 오디오 역시 파생 저작물에 해당하므로 재배포 시 원저작물 출처와 변경 내역을 함께 명시해야 하며, 상세 고지문은 평가셋 디렉터리마다 `ATTRIBUTION.md`로 들어 있다(`make_eval_t2.py`는 같은 꼴의 고지문을 직접 쓴다).

```text
This dataset contains audio derived from LibriSpeech (Panayotov et al., 2015),
distributed under CC BY 4.0. The derived multichannel audio was produced by
simulated room acoustics and is redistributed under the same license.
Source: https://www.openslr.org/12
```

### 외부 데이터셋 · 라이브러리 · 체크포인트 라이선스

| 자원 | 저장소 내 용도 | 라이선스 | 비고 |
|---|---|---|---|
| `LibriSpeech` (`dev-clean`) | 합성 평가셋 원천 음성, 화자 구분 시스템의 검증 음성 | `CC BY 4.0` | 파생 오디오 재배포 시 `ATTRIBUTION.md` 동봉 필수 |
| `LibriSpeech` (`test-clean`) | 본 평가셋(2인) 원천 음성 | `CC BY 4.0` | 파생 오디오 재배포 시 `ATTRIBUTION.md` 동봉 필수 |
| `LibriSpeech` (`train-clean-100`) | 화자 구분 시스템의 학습 음성 (학습 중에만 섞어 쓴다) | `CC BY 4.0` | 저장소에 포함하지 않는다 |
| `pyroomacoustics` (v0.8.5) | 다채널 실내 임펄스 응답(ISM) 생성 | `MIT` | 라이브러리 코드 미재배포 |
| `facebook/wav2vec2-base-960h` | 통제 베이스라인 1·2 및 제안 모델 동결 ASR 백엔드 | `Apache-2.0` | Hugging Face 런타임 로드 (가중치 미재배포) |
| `Qwen/Qwen3-ASR-1.7B` | 본 평가 공통 전사 백엔드 | `Apache-2.0` | 별도 qwen-asr 환경, 가중치 미재배포 |
| `nvidia/Nemotron-3-Diarization` | 최신 단일채널 화자 구분 비교군 | `OpenMDW-1.1` | 별도 Transformers 환경, 가중치 미재배포 |
| `microsoft/VibeVoice-ASR` | 단일채널 전사·화자 구분 비교군 | `MIT` | 별도 VibeVoice 환경, 가중치 미재배포 |
| `OpenMOSS-Team/MOSS-Transcribe-Diarize` | 단일채널 전사·화자 구분 비교군 | `Apache-2.0` | 별도 MOSS 환경, 가중치 미재배포 |
| `speechbrain/spkrec-ecapa-voxceleb` | 목소리 단서용 동결 화자 임베딩 백엔드 (목소리 전용·단순 결합·제안 모델) | `Apache-2.0` | 가중치 미재배포. `speechbrain` 1.1.2(`Apache-2.0`)로 런타임 로드하며, `transformers`와 한 환경에서 쓴다 |
| `snakers4/silero-vad` | 모듈형 파이프라인 음성 활동 검출 구성 요소 | `MIT` | 현재 통제 베이스라인은 내장 에너지 게이트 사용 |
| `transformers` | 사전학습 음향 백엔드 로더 | `Apache-2.0` | - |
| `torch` / `torchaudio` | 동결 ASR 모델 추론과 화자 구분 시스템 학습(`torch`). `torchaudio`는 저장소 코드가 임포트하지 않으며 `requirements.txt`에서 `torch`와 같은 마이너 버전으로 고정 | `BSD-3-Clause` / `BSD-2-Clause` | 동일 마이너 버전 조합 필요 |
| `numpy` / `scipy` | 음향 합성과 SRP-PHAT 신호처리(`numpy.fft`). `scipy`는 `pyroomacoustics`의 의존성이며 저장소 코드가 임포트하지 않음 | `BSD-3-Clause` | `evaluate.py`는 `numpy` 없이 표준 라이브러리만으로 동작 |
| `soundfile` | 오디오 읽기(`make eval-dev`의 `LibriSpeech` FLAC 등). WAV 쓰기는 표준 라이브러리 `wave` | `BSD-3-Clause` | 동적 링크되는 `libsndfile`은 `LGPL-2.1+` |

---

## 저장소 디렉터리 구조

```text
spec.py                      동결된 평가 계약 상수 및 스키마 정의 (수치·필드명의 단일 출처)
SPEC_DECISIONS.md            기획안 v1.7의 미정 항목을 spec.py에서 확정한 근거와 파급 범위
evaluate.py                  공식 채점기 (--pred / --gt / --track / --out)
simulate.py                  실내 음향 시뮬레이션 및 층화 설계 라이브러리 (--self-test 지원)
make_eval_dev.py             사전 측정 평가셋(eval-dev) 생성 스크립트 (--out / --n-sessions / --seed)
make_eval_t2.py              본 평가셋(2인) 생성 스크립트. 주 조건과 정지 조건 두 벌 (--out / --static-out / --librispeech-root)
baselines/single_channel.py  통제 베이스라인 1: 단일 채널 선택 + 동결 ASR + 균일 공간 분포
baselines/dsp_srp_phat.py    통제 베이스라인 2: SRP-PHAT 공간 분포 + 지연-합 빔포밍 + 동결 ASR
baselines/__init__.py        베이스라인 공통 로더, 인과적 청크 격자 및 JSONL 직렬화 유틸리티
data/eval-dev/               사전 측정 평가셋 (오디오·정답·메타·SHA-256 매니페스트·출처 고지)
data/eval-t2/                본 평가셋(2인) 주 조건: 턴마다 위치 이동 + 확산 잡음 (구성은 eval-dev 와 같다)
data/eval-t2-static/         본 평가셋(2인) 정지 조건: 같은 세션, 첫 위치·무잡음
results/                     통제 베이스라인 1·2의 공식 채점 결과 (baseline1.json, baseline2.json)
tests/test_evaluate.py       평가 계약 및 채점기 단위 테스트 모음 (153개 테스트)
tests/test_baselines.py      베이스라인 대표점 규칙 단위 테스트 (11개 테스트)
tests/test_eval_t2.py        본 평가셋 생성기 테스트 (가짜 음성, 7개 테스트)
tests/test_training.py       학습 코드 불변식 테스트 (torch 필요, 없으면 건너뛴다)
models/                      화자 구분 시스템: voice_only · space_only · naive_fusion · proposed(TAC + ECAPA + GRU),
                             공용 특징(common.py), TAC 블록(tac.py), 공용 온라인 슬롯 매칭(slots.py)
training/                    학습 코드: 설정(config.py), 음성 코퍼스(corpus.py), 방 음향 뱅크(rirbank.py),
                             교대 발화 세션 합성(scenes.py), GPU 위 방 음향 적용(render.py), 학습·검증 데이터(data.py),
                             손실(loss.py), 학습·검증 루프(engine.py), 공식 형식 예측·채점(official.py)
configs/                     시스템별 설정 (smoke, smoke_real, voice_only, space_only, naive_fusion, proposed_carry, proposed_reset)
train.py                     학습·검증 진입점 (--config / --data-root / --run-dir, 같은 run-dir 이면 이어서 학습)
modern_asr.py                Qwen 전사 및 최신 화자 구분 모델의 별도 환경 추론 작업자
mono_baselines.py            최신 단일채널 비교군의 재개·전사·채점 실행기
setup_asr_env.py             모델별 독립 환경 설치와 패키지 목록 보존
compare_asr.py               채널·관측 범위·평가 표본을 명시한 통합 비교표
tests/test_modern_asr.py     모델 API 어댑터·캐시 무결성·합성 오디오 종단 평가 테스트
notebooks/reproduce.ipynb    Baseline 2 추론·채점을 무료 CPU 런타임에서 재현하는 Colab 노트북
notebooks/train_template.ipynb  화자 구분 시스템을 Colab GPU 에서 학습·검증하는 템플릿 노트북 (CONFIG 하나만 바꾼다)
notebooks/mono_baselines.ipynb  최신 단일채널 비교군 3종의 Colab 추론·채점
notebooks/track_b_vibevoice.ipynb  같은 실행기의 VibeVoice 전용 프리셋
tools_make_demo_audio.py     웹페이지 6.2절 데모 오디오(단일 채널 · 정답 방위로 조향한 4채널)를 자르는 스크립트
docs/index.html              GitHub Pages 배포용 벤치마크 웹페이지 (제출 HTML과 동일 본)
requirements.txt             최소 채점용 및 전체 파이프라인용 의존성 명세
Makefile                     컴파일, 단위 테스트, 데이터 생성, 베이스라인 추론, 채점, 학습 자동화
```

코드 작성 원칙:

- **단일 상수 출처 유지**: `36`, `0.25`, `180.0`, `5.0`, `1e-12` 등 평가 계약과 직결되는 수치는 개별 모듈에 리터럴로 중복 기입하지 않고 `spec.py`에서 임포트해 사용한다.
- **무거운 의존성의 지연 임포트**: 채점·데이터 생성·베이스라인 모듈은 `torch`, `transformers`, `pyroomacoustics`, `soundfile` 등을 필요한 함수 내부에서 지연 임포트(lazy import)하여, 기본 파이썬 환경만으로 `make compile`이, 여기에 `numpy`만 더하면 `make test`가 항상 수행되도록 유지한다. 학습 코드(`models/`, `training/`)는 `torch`를 바로 임포트하지만, `make compile`은 모듈을 실행하지 않고 `make test`는 `torch`가 없으면 학습 테스트를 건너뛰므로 두 조건은 그대로다.

---

## 커밋 규약

커밋 제목은 무엇을 왜 바꾸는지 한국어 한 문장으로 쓴다.

```text
make clean 이 커밋된 데이터셋을 지우던 문제와 selftest 를 고친다
채널 수별 집계를 macro_per_case 로 맞추고 계약 공백 셋을 메운다
거리 대표점을 L1 최적 중앙값으로 바꾸고 재현 절차를 고친다
```

`spec.py`의 계약 상수를 변경할 때는 반드시 `SPEC_VERSION`을 상향하고 `SPEC_DECISIONS.md`에 변경 근거와 기존 점수에 미치는 영향을 함께 기록한다.
