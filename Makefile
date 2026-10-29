# Variable-Channel Spatial ASR benchmark -- task runner.
#
# GNU Make 3.81 (macOS 기본) 에서 동작하도록 작성했다. .ONESHELL 은 3.82+
# 에서만 지원하므로 쓰지 않는다. 문법 확인: `make -n compile`
#
# 모든 경로는 이 Makefile 이 있는 디렉터리 기준 상대경로다 (경로에 공백이
# 있어도 안전하도록 recipe 안에서 cd 하지 않는다).
#
# 아래 변수는 모두 커맨드라인에서 덮어쓸 수 있다.
#   make eval-dev N_SESSIONS=8
#   make baseline BASELINE=2
#   make score TRACK=1

# 주의: Make 는 `VAR = v  # c` 에서 `#` 앞 공백을 값에 포함시킨다.
# 그래서 설명 주석은 반드시 대입문 위에 따로 쓴다.
PY          ?= python3

# spec.MAX_SPEAKERS_PER_TRACK 의 키. 이번 회차 공식 범위는 Track 1 뿐이다.
# 어느 트랙을 돌릴지는 프로젝트 선택이므로 spec 에서 끌어오지 않는다.
TRACK       ?= 1
# 관측 범위. 하드코딩 금지 규칙에 따라 리터럴 "causal" 을 쓰지 않고
# spec.OFFICIAL_SCOPES[0] 에서 끌어온다. `?=` 라서 재귀 변수이고,
# $(SCOPE) 를 실제로 쓰는 타깃에서만 python 이 한 번 돈다.
SCOPE       ?= $(shell $(PY) -c 'import spec; print(spec.OFFICIAL_SCOPES[0])')
# 1 = 단일채널 동결 ASR + 균등 공간분포, 2 = GCC-PHAT / SRP-PHAT 고전 방식
BASELINE    ?= 1
# README "eval-dev 배포 규격" 의 500MB 상한 안쪽 (150 x 2.56MB = 384MB)
N_SESSIONS  ?= 150
SEED        ?= 20261029

DATA_DIR    ?= data/eval-dev
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
#   오프라인 스모크: make eval-dev EVAL_DEV_ARGS="--synthetic --rir-engine fallback_ism"
#                   make baseline BASELINE_ARGS="--synthetic-asr"
# 공식 수치에는 쓰지 마라. --synthetic / fallback_ism / --synthetic-asr 는 산출물에
# official=false 를 찍거나 전사를 가짜로 만든다.
EVAL_DEV_ARGS ?=
BASELINE_ARGS ?=

# doctest 를 돌릴 파일 목록. baselines/*.py 를 빼먹으면 173 개 예제가
# 조용히 안 돌아간다.
PY_DOCTEST_FILES = $(wildcard $(EVAL_ENTRY) $(SIM_ENTRY) baselines/*.py) make_eval_dev.py
BASELINE_MODS_PRESENT = $(patsubst baselines/%.py,baselines.%,$(filter-out baselines/__init__.py,$(wildcard baselines/*.py)))

.PHONY: help eval-dev eval-dev-verify sim-selftest baseline score test selftest compile clean distclean

# help: 타깃 목록을 출력한다 (기본 타깃).
help:
	@echo "targets:"
	@echo "  make compile      - 모든 파이썬 파일을 바이트컴파일해 문법 오류를 잡는다"
	@echo "  make test         - spec doctest + 모듈 doctest + tests/ 유닛테스트 (녹색이어야 하는 게이트)"
	@echo "  make selftest     - simulate / 베이스라인의 종단 불변식 자체점검 (무겁다)"
	@echo "  make eval-dev     - eval-dev 오디오와 정답 JSONL 을 $(DATA_DIR) 에 생성한다"
	@echo "  make eval-dev-verify - 렌더링 없이 eval-dev 불변식만 확인한다"
	@echo "  make sim-selftest - simulate.py 의 plan-layer 불변식만 돌린다"
	@echo "  make baseline     - 베이스라인 $(BASELINE) 추론을 돌려 $(PRED) 를 만든다"
	@echo "  make score        - $(PRED) 를 공식 지표로 채점해 $(SCORES) 를 만든다"
	@echo "  make clean        - 생성물(데이터·예측·점수·바이트코드)을 지운다"

# eval-dev: 고정 시드로 평가용 다채널 오디오 + 정답 라벨을 합성한다 (네트워크 불필요).
#
# make_eval_dev.py 에는 --track 플래그가 없다. eval-dev 는 Track 1 구성이고,
# 트랙 선택은 추론·채점 쪽(baseline / score)에서만 의미가 있다.
# 공식 산출물은 LibriSpeech + pyroomacoustics 가 필요하다. 둘이 없는 환경에서는
# EVAL_DEV_ARGS="--synthetic --rir-engine fallback_ism" 로 스모크만 돌 수 있고,
# 그 산출물에는 official=false 가 찍힌다.
eval-dev:
	$(PY) $(GEN_ENTRY) --out $(DATA_DIR) --n-sessions $(N_SESSIONS) --seed $(SEED) $(EVAL_DEV_ARGS)

# eval-dev-verify: 오디오를 렌더링하지 않고 plan-layer 불변식만 확인한다 (numpy 불필요).
eval-dev-verify:
	$(PY) $(GEN_ENTRY) --out $(DATA_DIR) --n-sessions $(N_SESSIONS) --seed $(SEED) --verify-only

# sim-selftest: simulate.py 가 지금 실제로 제공하는 CLI. 데이터는 쓰지 않는다.
#
# N_SESSIONS 를 넘기지 않는다. 그건 eval-dev 데이터셋 크기이고, 자체점검의
# 세션 수는 층화(stratification) 제약을 받는 다른 값이다 -- 나누어떨어지지
# 않는 수를 주면 simulate.py 가 ValueError 로 거부한다 (실제로 6 에서 거부됨).
# 세션 수를 바꾸려면 SIM_SESSIONS 를 쓰고, 거부되면 그 수를 쓰지 마라.
SIM_SESSIONS ?=
sim-selftest:
	$(PY) $(SIM_ENTRY) --self-test --seed $(SEED) $(if $(SIM_SESSIONS),--n-sessions $(SIM_SESSIONS),)

# baseline: 학습이 필요 없는 베이스라인으로 제출 포맷 그대로의 예측 JSONL 을 뽑는다.
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
# clean 이 지우면 358 MB 와 CC BY 4.0 표기 의무 파일이 같이 날아간다.
# 데이터셋을 지우려면 distclean 을 명시적으로 부른다.
clean:
	rm -rf $(PRED) $(SCORES) scores_*.json runs/
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

distclean: clean
	@echo "이 명령은 $(DATA_DIR) 를 통째로 지운다 (커밋된 평가셋 + ATTRIBUTION.md)."
	@echo "되돌리려면: git checkout -- $(DATA_DIR)   또는   make eval-dev SEED=$(SEED)"
	@printf '계속하려면 yes 를 입력: ' && read ans && [ "$$ans" = yes ]
	rm -rf $(DATA_DIR)
