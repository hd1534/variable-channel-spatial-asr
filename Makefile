# Variable-Channel Spatial ASR benchmark -- task runner.
#
# GNU Make 3.81 (macOS 기본) 에서 동작하도록 작성했다. .ONESHELL 은 3.82+
# 에서만 지원하므로 쓰지 않는다. 문법 확인: `make -n compile`
#
# 모든 경로는 이 Makefile 이 있는 디렉터리 기준 상대경로다 (경로에 공백이
# 있어도 안전하도록 recipe 안에서 cd 하지 않는다).
#
# 아래 변수는 모두 커맨드라인에서 덮어쓸 수 있다.
#   make eval-dev LIBRISPEECH_ROOT=/path/to/LibriSpeech/dev-clean
#   make baseline BASELINE=2
#   make score TRACK=1

# 주의: Make 는 `VAR = v  # c` 에서 `#` 앞 공백을 값에 포함시킨다.
# 그래서 설명 주석은 반드시 대입문 위에 따로 쓴다.
PY          ?= python3

# spec.MAX_SPEAKERS_PER_TRACK 의 키. 1 = 단일 화자(사전 측정), 2 = 세션당 2인.
# 공개된 eval-dev 가 단일 화자 사전 측정셋이라 기본값은 1 이다.
TRACK       ?= 1
# 관측 범위. 하드코딩 금지 규칙에 따라 리터럴 "causal" 을 쓰지 않고
# spec.OFFICIAL_SCOPES[0] 에서 끌어온다. `?=` 라서 재귀 변수이고,
# $(SCOPE) 를 실제로 쓰는 타깃에서만 python 이 한 번 돈다.
SCOPE       ?= $(shell $(PY) -c 'import spec; print(spec.OFFICIAL_SCOPES[0])')
# 1 = 단일채널 동결 ASR + 균등 공간분포, 2 = GCC-PHAT / SRP-PHAT 고전 방식
BASELINE    ?= 1
# 커밋된 eval-dev 의 세션 수. 방위 36구간·거리 8구간이 엄격히 균일해져 official=true 가
# 나오는 가장 작은 수다 (144 의 배수만 그렇고, 150 처럼 층화가 안 되는 수는 생성기가 거부한다).
N_SESSIONS  ?= 144
SEED        ?= 20261029

# 추론·채점이 읽는 평가셋. 저장소에 커밋된 공식 세트다.
DATA_DIR    ?= data/eval-dev
# make eval-dev 가 쓰는 곳. 커밋된 $(DATA_DIR) 와 일부러 다르게 둔다. 생성기는 계획에
# 없는 세션 파일을 지우고 정답·매니페스트를 새로 쓰기 때문이다 (생성기도 manifest 가
# official=true 인 디렉터리에는 --overwrite-official 없이 쓰지 않는다).
GEN_DIR     ?= data/eval-dev-rebuild
# make eval-dev 에 필요한 LibriSpeech dev-clean 루트 (<root>/<speaker>/<chapter>/*.flac).
LIBRISPEECH_ROOT ?=
# evaluate.py 의 실제 플래그는 --gt 이고, 모듈 docstring 이 문서화한 파일명은
# ground_truth.jsonl 이다 (확인: `python3 evaluate.py --help`).
GT          ?= $(DATA_DIR)/ground_truth.jsonl
PRED        ?= predictions.jsonl
SCORES      ?= scores.json

# 케이스 목록(= 베이스라인이 읽는 per-case 매니페스트)은 meta.jsonl 이다.
# 명시적으로 넘겨야 한다: 베이스라인의 자동탐색 후보 목록에 meta.jsonl 이 없고,
# 대신 데이터셋 요약 파일인 manifest.json 을 먼저 집어서 터진다
# (manifest.json 은 per-case 행이 아니라 files/hash/constants 요약이다).
MANIFEST    ?= $(DATA_DIR)/meta.jsonl

# 베이스라인은 패키지 모듈로 호출한다 (`python -m baselines.single_channel`).
# BASELINE=1 -> $(BASELINE1_MOD), BASELINE=2 -> $(BASELINE2_MOD) 로 전개된다.
# 둘 다 --eval-dir/--out/--scope/--track 을 받는 것을 --help 로 확인했다.
BASELINE1_MOD ?= baselines.single_channel
BASELINE2_MOD ?= baselines.dsp_srp_phat
BASELINE_MOD  ?= $(BASELINE$(BASELINE)_MOD)

# simulate.py 는 세션 설계·합성 라이브러리 + 자체점검이고, 데이터셋을 디스크에
# 쓰는 엔트리포인트는 make_eval_dev.py 다 (확인: 두 파일의 --help).
SIM_ENTRY     ?= simulate.py
GEN_ENTRY     ?= make_eval_dev.py
EVAL_ENTRY    ?= evaluate.py

# 타깃에 추가 인자를 그대로 넘기는 통로.
#   make eval-dev EVAL_DEV_ARGS="--quiet"
#   make baseline BASELINE_ARGS="--limit 12"
# 코퍼스·체크포인트 없이 전 경로를 돌려 보려면 이 통로 대신 make smoke 를 쓴다.
# --synthetic / fallback_ism / --synthetic-asr 는 산출물에 official=false 를 찍거나
# 전사를 가짜로 만들므로 공식 수치에는 쓰지 마라.
EVAL_DEV_ARGS ?=
BASELINE_ARGS ?=

# doctest 를 돌릴 파일 목록. baselines/*.py 를 빼먹으면 그 안의 doctest 예제가
# 조용히 안 돌아간다.
PY_DOCTEST_FILES = $(wildcard $(EVAL_ENTRY) $(SIM_ENTRY) baselines/*.py) make_eval_dev.py
BASELINE_MODS_PRESENT = $(patsubst baselines/%.py,baselines.%,$(filter-out baselines/__init__.py,$(wildcard baselines/*.py)))

.PHONY: help eval-dev eval-dev-check eval-dev-verify sim-selftest asr-cache baseline score smoke test selftest compile clean distclean

# help: 타깃 목록을 출력한다 (기본 타깃).
help:
	@echo "targets:"
	@echo "  make compile      - 모든 파이썬 파일을 바이트컴파일해 문법 오류를 잡는다"
	@echo "  make test         - spec doctest + 모듈 doctest + tests/ 유닛테스트 (녹색이어야 하는 게이트)"
	@echo "  make selftest     - simulate / 베이스라인의 종단 불변식 자체점검 (무겁다)"
	@echo "  make smoke        - 합성 데이터로 생성 -> 추론 -> 채점을 $(SMOKE_DIR) 안에서 한 번 돈다"
	@echo "  make asr-cache    - 베이스라인의 동결 ASR 체크포인트를 한 번 내려받는다 (네트워크 필요)"
	@echo "  make baseline     - $(DATA_DIR) 에 베이스라인 $(BASELINE) 추론을 돌려 $(PRED) 를 만든다"
	@echo "  make score        - $(PRED) 를 공식 지표로 채점해 $(SCORES) 를 만든다"
	@echo "  make eval-dev     - eval-dev 를 $(GEN_DIR) 에 다시 만든다 (LIBRISPEECH_ROOT 필요)"
	@echo "  make eval-dev-check - $(GEN_DIR) 의 오디오·정답·메타가 커밋된 $(DATA_DIR) 와 바이트 단위로 같은지 확인한다"
	@echo "  make eval-dev-verify - 렌더링 없이 eval-dev 불변식만 확인한다"
	@echo "  make sim-selftest - simulate.py 의 plan-layer 불변식만 돌린다"
	@echo "  make clean        - 예측·점수·runs/·바이트코드를 지운다 (평가셋은 그대로 둔다)"

# eval-dev: 고정 시드로 사전 측정 평가셋(단일 화자)을 $(GEN_DIR) 에 다시 합성한다
# (네트워크 불필요). 커밋된 $(DATA_DIR) 에는 쓰지 않는다. 같은 시드·세션 수로 만들면
# 오디오·정답·메타가 커밋된 세트와 바이트 단위로 같아야 하고, make eval-dev-check 로
# 확인한다 (정답은 FLAC 을 코퍼스 기준 상대 경로로 적으므로 코퍼스를 푼 위치와 무관하다).
# 다시 만든 $(GEN_DIR) 도 official=true 라서 생성기가 덮어쓰기를 거부한다. 다시 돌리려면 먼저 지운다.
#
# make_eval_dev.py 에는 --track 플래그가 없다. eval-dev 는 단일 화자 구성이고,
# 트랙 선택은 추론·채점 쪽(baseline / score)에서만 의미가 있다.
# 공식 산출물은 LibriSpeech dev-clean + pyroomacoustics 가 필요하다. 코퍼스 없이
# 전 경로만 확인하려면 make smoke 를 쓴다.
eval-dev:
	$(PY) $(GEN_ENTRY) --out $(GEN_DIR) --n-sessions $(N_SESSIONS) --seed $(SEED) \
	  $(if $(LIBRISPEECH_ROOT),--librispeech-root "$(LIBRISPEECH_ROOT)",) $(EVAL_DEV_ARGS)

# eval-dev-check: make eval-dev 로 다시 만든 $(GEN_DIR) 의 오디오·정답·메타를 커밋된
# $(DATA_DIR) 와 바이트 단위로 비교한다. rir/ 는 커밋하지 않고, manifest.json 은 생성
# 시각과 코퍼스 경로가 들어가며, ATTRIBUTION.md 는 생성기가 쓰지 않으므로 뺀다.
eval-dev-check:
	diff -rq --exclude=rir --exclude=manifest.json --exclude=ATTRIBUTION.md $(DATA_DIR) $(GEN_DIR)
	@echo "$(GEN_DIR) 의 오디오·정답·메타가 $(DATA_DIR) 와 같다"

# eval-dev-verify: 오디오를 렌더링하지 않고 plan-layer 불변식만 확인한다 (numpy 불필요).
# --verify-only 는 아무것도 쓰지 않는다.
eval-dev-verify:
	$(PY) $(GEN_ENTRY) --out $(GEN_DIR) --n-sessions $(N_SESSIONS) --seed $(SEED) --verify-only

# sim-selftest: simulate.py 가 지금 실제로 제공하는 CLI. 데이터는 쓰지 않는다.
#
# N_SESSIONS 를 넘기지 않는다. 그건 eval-dev 데이터셋 크기이고, 자체점검의
# 세션 수는 층화(stratification) 제약을 받는 다른 값이다 -- 나누어떨어지지
# 않는 수를 주면 simulate.py 가 ValueError 로 거부한다 (실제로 6 에서 거부됨).
# 세션 수를 바꾸려면 SIM_SESSIONS 를 쓰고, 거부되면 그 수를 쓰지 마라.
SIM_SESSIONS ?=
sim-selftest:
	$(PY) $(SIM_ENTRY) --self-test --seed $(SEED) $(if $(SIM_SESSIONS),--n-sessions $(SIM_SESSIONS),)

# asr-cache: 베이스라인이 쓰는 동결 ASR 체크포인트(baselines.DEFAULT_ASR_MODEL,
# 약 360 MiB)를 한 번 내려받아 Hugging Face 캐시에 둔다. 베이스라인은
# HF_HUB_OFFLINE=1 과 local_files_only 로 고정되어 스스로 내려받지 않으므로, 새
# 환경에서는 make baseline 전에 한 번 돌린다. 네트워크를 쓰는 타깃은 이것뿐이다.
asr-cache:
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 $(PY) -c "import baselines; from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; m = baselines.DEFAULT_ASR_MODEL; Wav2Vec2Processor.from_pretrained(m); Wav2Vec2ForCTC.from_pretrained(m); print('cached', m)"

# baseline: 학습이 필요 없는 베이스라인으로 제출 포맷 그대로의 예측 JSONL 을 뽑는다.
# 새 환경이면 make asr-cache 를 먼저 돌린다 (--synthetic-asr 는 체크포인트가 필요 없다).
baseline:
	@$(PY) -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('$(BASELINE_MOD)') else 1)" 2>/dev/null \
	  || { echo "ERROR: BASELINE=$(BASELINE) -> 모듈 $(BASELINE_MOD) 가 없다."; \
	       echo "  현재 있는 베이스라인 모듈: $(BASELINE_MODS_PRESENT)"; exit 1; }
	$(PY) -m $(BASELINE_MOD) --eval-dir $(DATA_DIR) --out $(PRED) --scope $(SCOPE) --track $(TRACK) \
	  $(if $(wildcard $(MANIFEST)),--manifest $(MANIFEST),) $(BASELINE_ARGS)

# score: 공식 평가기를 돌린다. spec.py 가 유일한 상수 출처이므로 인자로 임계값을 넘기지 않는다.
# 관측 범위는 CLI 플래그가 아니다. evaluate.py 가 레코드의 observation_mode 를 읽고
# spec.OFFICIAL_SCOPES 로 직접 공식/비공식 표를 나눈다.
score:
	$(PY) $(EVAL_ENTRY) --pred $(PRED) --gt $(GT) --track $(TRACK) --out $(SCORES)

# smoke: 코퍼스·체크포인트 없이 생성 -> 추론 -> 채점 전 경로를 한 번 돈다 (네트워크
# 불필요). 산출물은 모두 $(SMOKE_DIR) 아래에 쓰고 커밋된 $(DATA_DIR) 는 건드리지 않는다.
# 합성 음성·fallback_ism·--synthetic-asr 산출물이라 official=false 이고 점수는 의미가 없다.
SMOKE_DIR      ?= runs/smoke
SMOKE_SESSIONS ?= 4
smoke:
	$(PY) $(GEN_ENTRY) --out $(SMOKE_DIR)/eval-dev --n-sessions $(SMOKE_SESSIONS) --seed $(SEED) \
	  --synthetic --rir-engine fallback_ism --quiet
	$(PY) -m $(BASELINE_MOD) --eval-dir $(SMOKE_DIR)/eval-dev --out $(SMOKE_DIR)/predictions.jsonl \
	  --scope $(SCOPE) --track $(TRACK) --manifest $(SMOKE_DIR)/eval-dev/meta.jsonl --synthetic-asr
	$(PY) $(EVAL_ENTRY) --pred $(SMOKE_DIR)/predictions.jsonl --gt $(SMOKE_DIR)/eval-dev/ground_truth.jsonl \
	  --track $(TRACK) --out $(SMOKE_DIR)/scores.json > /dev/null
	@echo "smoke 통과: $(SMOKE_DIR)/scores.json (official=false, 점수는 의미 없음)"

# test: 무거운 의존성 없이 stdlib + numpy 만으로 통과해야 하는 테스트 모음.
test: compile
	$(PY) spec.py
	@for m in $(PY_DOCTEST_FILES); do \
	  if [ -f "$$m" ]; then echo "doctest $$m"; $(PY) -m doctest "$$m" || exit 1; fi; \
	done
	@if [ -d tests ]; then \
	  if [ -f tests/__init__.py ]; then top=.; else top=tests; fi; \
	  echo "unittest discover -t $$top"; \
	  $(PY) -m unittest discover -s tests -t $$top -p 'test_*.py' -v; \
	else \
	  echo "tests/ 아직 없음 -- 유닛테스트 디스커버리 건너뜀"; \
	fi

# selftest: 종단 불변식 점검. make test 보다 무겁고, 합성 오디오를 임시 디렉터리에
# 만들어 CLI 전체 경로를 한 번 돈다 (네트워크·체크포인트 불필요, --synthetic-asr).
selftest:
	$(PY) $(SIM_ENTRY) --self-test
	@for mod in $(BASELINE_MODS_PRESENT); do \
	  echo "== $$mod --self-test"; \
	  $(PY) -m $$mod --self-test || exit 1; \
	done

# compile: 저장소의 모든 파이썬 파일을 바이트컴파일한다. 지연 임포트 규칙이 지켜지면 의존성 없이 통과한다.
compile:
	$(PY) -m compileall -q spec.py make_eval_dev.py $(wildcard $(EVAL_ENTRY) $(SIM_ENTRY)) $(wildcard baselines/*.py tests/*.py)

# clean: 임시 산출물만 지운다. 커밋된 eval-dev 는 건드리지 않는다 (distclean 참조).
# data/eval-dev 는 저장소에 커밋된 공식 평가셋이고 ATTRIBUTION.md 가 그 안에 있다.
# clean 이 지우면 357.5 MiB 와 CC BY 4.0 표기 의무 파일이 같이 날아간다.
# 데이터셋을 지우려면 distclean 을 명시적으로 부른다. make eval-dev 로 다시 만든
# $(GEN_DIR) 도 지우지 않는다 (필요 없으면 직접 지운다).
clean:
	rm -rf $(PRED) $(SCORES) scores_*.json runs/
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

distclean: clean
	@echo "이 명령은 $(DATA_DIR) 를 통째로 지운다 (커밋된 평가셋 + ATTRIBUTION.md)."
	@echo "되돌리려면: git checkout -- $(DATA_DIR)"
	@printf '계속하려면 yes 를 입력: ' && read ans && [ "$$ans" = yes ]
	rm -rf $(DATA_DIR)
