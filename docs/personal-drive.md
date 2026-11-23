# 개인 Drive를 공식 Python API로 연결하기

학교 계정의 Colab Pro+에서 실행하고, 개인 계정의 Drive 용량으로 백업한다.
Google 공식 `google-api-python-client`, `google-auth`, `google-auth-oauthlib`를 사용하며
인증은 사용자가 만든 OAuth 프로젝트에 연결된다. 공용 rclone 프로젝트에 의존하지 않는다.
이 방식은 파일시스템 마운트가 아니다. 모델은 Colab 로컬 파일을 사용한다.

## 1. 개인 OAuth 클라이언트 만들기

개인 Google 계정으로 [Google Cloud Console](https://console.cloud.google.com/)에 로그인한다.

1. 프로젝트 선택 메뉴에서 이 백업 전용 프로젝트를 만든다. 예: `vcsasr-backup`.
2. **API 및 서비스 → 라이브러리 → Google Drive API → 사용**을 선택한다.
3. **Google Auth Platform → Branding**에서 앱 이름, 지원 이메일, 연락처 이메일을 입력한다.
4. **Audience**는 개인 Gmail 계정을 사용할 수 있도록 **External**로 설정한다.
   Testing으로 시험할 경우 **Test users**에 인증할 개인 계정을 추가한다.
5. **Data Access**에서 `https://www.googleapis.com/auth/drive.file` 범위를 추가한다.
   이 앱이 만든 백업 파일을 대상으로 하며, 기존 Drive 전체를 읽는 권한은 요청하지 않는다.
6. **Clients → Create client → Desktop app**을 선택하고 JSON을 다운로드한다.
7. 다운로드한 파일을 프로젝트의 `runs/private/credentials.json`에 둔다.
   이 파일은 Git에 포함하지 않으며, 채팅이나 노트북 출력에 붙여 넣지 않는다.

공식 절차: [Python quickstart](https://developers.google.com/workspace/drive/api/quickstart/python),
[Drive API 권한 범위](https://developers.google.com/workspace/drive/api/guides/api-specific-auth).

**장기 사용 시 게시 상태:** External 앱이 **Testing** 상태이면 Drive 권한의 갱신 토큰은 7일 뒤 만료된다.
처음 시험한 뒤 장기간 사용하려면 **Audience → Publish app**에서 In production으로 전환하고,
아래 인증을 `--reauthorize`로 다시 진행한다. Google이 표시하는 앱 설정·검증 요구사항을 따른다.
게시 상태와 관계없이 사용자가 권한을 취소하면 다시 인증해야 한다.
[Google 토큰 만료 안내](https://developers.google.com/identity/protocols/oauth2#expiration).

## 2. Mac에서 개인 계정 인증하기

`vcsasr` 디렉터리의 터미널에서 실행한다. ASR 패키지와 분리하려면 `.venv-drive` 같은 별도 venv를 사용해도 된다.

```bash
python3 -m pip install -r requirements-drive.txt
python3 setup_drive.py
```

열린 브라우저에서 **5TB 용량을 가진 개인 계정**을 선택해 승인한다.
localhost 콜백으로 인증을 받으므로 인증 코드를 수동으로 복사하지 않는다.

- 개인 Drive에 전용 `vcsasr-backups` 폴더를 만들고 링크를 출력한다.
- `runs/private/drive-auth.json`에 갱신 토큰·개인 OAuth 클라이언트·폴더 ID를 저장한다. 파일 권한은 `0600`이다.
- 토큰 자체는 출력하지 않는다. 이 JSON 파일은 계정 접근 권한이므로 비공개로 보관한다.

인증할 계정을 바꾸거나 새 토큰이 필요하면 다음을 실행한다.

```bash
python3 setup_drive.py --reauthorize
```

같은 계정·같은 OAuth 클라이언트이면 기존 백업 폴더를 재사용한다.
계정이나 클라이언트가 바뀌면 새 폴더를 만들고 이전 백업은 남겨 둔다.
인증 파일을 다른 곳에 보관하려면 `--client-secrets`와 `--out`을 지정한다.

## 3. 학교 계정 Colab에 비공개 인증 등록하기

1. 학교 Pro+ 계정으로 프로젝트 노트북을 연다.
2. 왼쪽 **열쇠 아이콘 → 보안 비밀**에서 `DRIVE_AUTH`라는 이름을 만든다.
3. 로컬 `runs/private/drive-auth.json`의 전체 JSON을 값으로 넣는다.
4. 사용할 각 노트북에 접근을 허용한다.

인증 정보를 코드 셀, 출력, Git, 채팅에 복사하지 않는다.
노트북 설정의 기본값은 다음과 같다.

```python
STORAGE_BACKEND: str = "drive_api"
DRIVE_SECRET: str = "DRIVE_AUTH"
LOCAL_STORE_ROOT: str = "/content/vcsasr-store"
BACKUP_EVERY_SEC: int = 120
```

연결 셀은 공식 라이브러리를 설치하고, 백업을 복원한 다음 실제 쓰기를 확인한다.
이미 복원한 VM에서 다시 실행하면 처음부터 받지 않고 다른 런타임의 변경만 받아 동기화한다.
이 단계가 성공해야 모델 준비·학습을 시작한다. 연결 셀이 실패하면 이후 명령 셀은 백업 없이 실행하지 않고 오류로 멈춘다.
Google API 요청 한도는 개인 프로젝트에도 적용되며,
일시적인 요청 제한, 서버 오류, 네트워크 끊김(DNS 실패, 응답 끊김, 토큰 갱신 중 오류 포함)은 약 3분 동안 지수 백오프로 재시도한다.
용량 부족·권한 취소·API 미활성화는 성공으로 숨기지 않는다.

개인 계정에서 Colab과 Drive를 함께 사용한다면 `STORAGE_BACKEND="mounted_drive"`로
기존 `google.colab.drive.mount()`를 선택할 수 있다. 이 경우 개인 OAuth 설정은 필요 없다.

## 4. 백업 완료와 복원 기준

- `runs/`, 평가 결과, RIR, 전사 캐시는 120초마다 백업한다. 주기 외에 전송 시간도 필요하다.
- 대용량 `models/`, `cache/`는 외부 명령이 끝날 때 백업한다. 그 전에 작은 학습 결과부터 확정한다.
- 전송은 청크 단위로 수행하고, 업로드가 끝나면 Drive의 **크기·MD5**를 확인한다.
- 모든 파일 전송이 성공한 뒤에만 완료 기록을 게시한다. 복원 때 **SHA-256**도 검증한다.
- 복원이 중간에 끊기면 다시 실행할 때 이미 받은 파일은 다시 받지 않고 이어 간다. 그 사이 다른 런타임이 백업했어도 같다.
- 명령 종료와 마지막 셀은 백업이 끝날 때까지 기다린다. 완료 표시는 마지막 셀의 `개인 Drive 최종 백업 완료` 줄이다.
  그 앞의 로그에는 새로 올린 파일이 있으면 `[storage] backup complete: N files (commit …)`,
  새 파일이 없으면 `[storage] backup up to date`가 나온다.
- 백업 오류는 실행 중인 작업에 전달하며, 로컬 파일을 보존한다. 오류가 해소되면 연결 셀부터 다시 실행한다.

**갑작스러운 Colab VM 종료 전 아직 백업하지 못한 작업은 유실될 수 있다.**
공식 API를 사용해도 아직 전송하지 않은 데이터를 복구할 수는 없다.
네트워크가 잠깐 끊긴 경우 살아 있는 프로세스 안에서 청크 업로드를 재시도한다.
VM이 초기화되어 업로드 세션을 잃으면 마지막 완료 백업을 복원하고 필요한 파일을 다시 전송한다.

백업 폴더에는 변경 불가능한 `objects/`와 `commits/`가 보인다. 웹에서 원래 `runs/` 트리가 보이는 형식은 아니다.
동일 내용은 중복 전송하지 않지만, 이전 체크포인트의 원격 이력은 자동 삭제하지 않는다.
따라서 장기간 실험하면 최신 로컬 결과보다 큰 공간을 사용할 수 있다.

## 5. 여러 런타임에서 같은 폴더 쓰기

여러 런타임이 같은 백업 폴더를 동시에 써도 된다. 예를 들어 학습 노트북(`train_template.ipynb`)과 본 평가 노트북(`benchmark.ipynb`)을
서로 다른 런타임에서 같은 `DRIVE_AUTH`로 실행할 수 있다. 단, 모든 런타임이 새 코드를 돌릴 때만 그렇다(아래 "이전 코드에서 올릴 때").

- 런타임마다 자기 백업 이력을 이어 간다. 다른 런타임이 백업했다고 실행 중인 작업이 멈추지 않는다.
- 다음 동기화 때 여러 이력을 합친다(`[storage] merged K backup histories`).
  동기화는 연결 셀과 노트북 명령(`storage.run`)이 시작할 때마다 한다.
  다른 런타임에서 바뀐 파일은 이때 받는다(`[storage] updating N files from other runtimes`).
- 합치기는 git의 recursive merge처럼 두 이력씩 실제 공통 조상을 기준으로 한다. 그래서 공통 조상 이후 한 런타임만 바꾼
  실행·평가 폴더(`runs/<실행>`, `eval/<이름>`, `mono_baselines/<실행>`, `rirbanks/<뱅크>` 등)는 런타임이 몇 개이든,
  앞서 몇 번 합쳤든 언제나 그 런타임의 버전을 쓴다.
- 공통 조상 이후 두 런타임이 모두 바꾼 폴더만 충돌이다. 충돌하면 더 나중에 쓴 버전을 쓰고, 다른 버전은 백업 이력에 남긴다.
  버전마다 백업이 그 폴더를 바꾼 시각(가장 늦게 쓴 파일의 수정 시각이되, 늘 바꾸기 전 버전보다 늦은 값)을 기록해 비교하므로,
  파일을 지우기만 한 버전도 그 전 버전보다 새것으로 친다.
  로그 `[storage] <폴더> changed in two runtimes; kept the most recently written version (commit …); the other version stays in the backup history (commit …)`의
  두 번째 커밋 ID를 7절의 `export --commit`에 주면 그 버전을 꺼낼 수 있다.
- 한 런타임이 지운 폴더를 다른 런타임이 지운 버전보다 나중에 바꿨다면 바꾼 버전을 남긴다. 다른 런타임의 버전이
  지운 버전보다 새것이 아니면 지운 상태를 따른다. 어느 쪽이든 로그에 남기고 다른 버전은 백업 이력에 남긴다.
- ASR 전사 캐시 폴더(`ASR_CACHE_NAME`, 기본 `asr_cache/`)는 `<백엔드>.jsonl`, `<백엔드>.inputs.json`,
  `qwen-runtime.json`을 한 단위로 합친다. 두 런타임이 기록한 실행 계약(`qwen-runtime.json`)이 같고
  `<백엔드>.inputs.json`에서 세션 오디오 해시가 어긋나지 않으면 전사와 출처 기록을 합친다.
  그렇지 않으면 섞지 않고, 위 충돌 규칙대로 파일을 더 늦게 쓴 쪽의 폴더 전체를 쓰고 충돌을 로그에 남긴다.
  다른 버전은 백업 이력에 남는다.

그래도 **같은 설정을 두 런타임에서 동시에 학습하지는 않는다.** GPU 시간을 두 번 쓰고,
한쪽 결과는 최신 버전에서 빠져 이력에만 남는다.

**이전 코드에서 올릴 때:** 이전 커밋(`9123e53`)으로 시작한 런타임은 이전 저장 코드를 그대로 돌린다.
이 코드는 다른 런타임이 백업을 올리면 다음 주기 백업에서 `remote advanced in another runtime` 오류로 작업을 멈춘다.
새 코드의 연결 셀은 연결하자마자 백업을 올린다. 그래서 새 코드로 두 번째 런타임을 연결하기 전에
이전 코드로 돌고 있는 런타임을 모두 멈추거나 새 코드로 다시 실행한다.

### 복원 제외 경로 `RESTORE_SKIP`

`RESTORE_SKIP`은 복원·동기화 때 그 런타임의 VM으로 내려받지 않을 경로 접두어 목록이다.
건너뛴 파일은 받지 않을 뿐 백업에서 지워지지 않고, 그 런타임의 백업 뒤에도 그대로 남는다.
노트북을 그대로 두면 다음 기본값을 쓴다.

| 노트북 | 기본 `RESTORE_SKIP` |
|---|---|
| `train_template.ipynb` | 본 평가 결과(`eval`, `mono_baselines`), 전사 캐시(`asr_cache`), 공개 모델 가중치(`models/asr`, `models/huggingface`) |
| `benchmark.ipynb` | `cache`, `rirbanks`, 학습 재개용 체크포인트 `runs/<실행>/ckpt`, 공개 모델 가중치(`models/asr`, `models/huggingface`). 우리 시스템을 고르지 않으면 `runs`, `models/ecapa`도 |

공개 모델 가중치는 revision을 고정했으므로 Hugging Face에서 다시 받아도 같은 파일이다.
그래서 `models/asr`와 `models/huggingface`는 **백업하지도 복원하지도 않는다**(`drive_storage.PUBLIC_WEIGHT_DIRS`).
- 새로 받은 가중치는 Drive에 올리지 않는다.
- 이 규칙 전에 백업에 들어간 가중치는 기록에 그대로 남는다. `RESTORE_SKIP = []`로 연결하면 그 파일은 받을 수 있다.

전송은 요청 하나에 1 GiB씩 오간다(`drive_api.CHUNK_BYTES`). 처음에는 8 MB였는데, 요청마다 왕복 시간이 붙어서 Colab에서 큰 파일 복원이 초당 수 MB에 그쳤다.
Drive는 저장한 파일마다 MD5·SHA-1·SHA-256을 직접 계산해 둔다. 백업 객체 이름이 내용의 SHA-256이라서 이 값으로 확인하면 파일을 다시 읽지 않아도 된다.
- 업로드: 바뀐 파일을 찾을 때 이미 계산한 SHA-256을 Drive의 `sha256Checksum`과 비교한다.
- 다운로드: 받으면서 MD5를 계산해 Drive 값과 맞춰 보고, SHA-256은 Drive가 보고한 값으로 확인한다.
- Drive가 SHA-256을 보고하지 않을 때만 로컬에서 파일을 한 번 더 읽어 확인한다.

다른 목록이 필요하면 연결 셀 전에 `RESTORE_SKIP`에 경로 접두어 목록을 넣는다. 빈 목록(`[]`)이면 모두 받는다.

## 6. 학교 Drive 결과 가져오기

학교 Drive의 기존 파일은 자동으로 옮기거나 지우지 않는다. 가져오지 않으면 첫 `drive_api` 실행은 빈 백업에서 시작하므로
모든 시스템을 처음부터 학습하고 RIR 뱅크와 LibriSpeech 캐시도 새로 만든다. 기존 결과는 한 번만 가져오면 된다.

**학교 계정 Colab에서**: 학습 노트북의 `IMPORT_FROM`에 학교 Drive의 결과 경로를 넣고 실행한다.

```python
IMPORT_FROM: str = "/content/drive/MyDrive/vcsasr"
```

연결 셀이 학교 Drive를 `/content/drive`에 마운트하고, 개인 백업에 아직 없는 폴더
(`runs/<실행>`, `eval/<이름>`, `rirbanks/<뱅크>`, `cache/` 파일 등)만 `LOCAL_STORE_ROOT`로 복사한 뒤 백업한다.
백업에 이미 있는 폴더는 덮어쓰거나 섞지 않는다. 단, ASR 전사 캐시 폴더는 5절의 합치기 규칙대로 합친다.
실행 계약이나 세션 오디오 해시가 다르면 섞지 않고 백업의 캐시를 그대로 둔다
(`[storage] kept this backup's asr_cache; ...`). 가져온 뒤에는 `IMPORT_FROM`을 비워 둔다.

**Mac에서**: 학교 결과의 사본이 있으면 다음으로 가져온다. `--local`에는 이 백업에 쓸 새 빈 폴더를 준다.

```bash
python3 drive_storage.py import \
  --auth runs/private/drive-auth.json \
  --from <학교 vcsasr 폴더 사본> \
  --local runs/import-workspace
```

두 방법 모두 다음과 같이 가져온다.

- 원본 폴더(합치기 단위)마다 한 백업에 한 번만 가져오고, 가져온 폴더를 백업에 기록한다. 그래서 `IMPORT_FROM`을
  그대로 두고 다시 실행해도, 가져온 뒤 이름을 바꾸거나 지운 폴더를 다시 들여오지 않는다.
  예를 들어 학습 노트북이 이전 `eval/<실행>`을 `eval/<실행>-<ASR>`로 옮긴 뒤에도 `eval/<실행>`이 다시 생기지 않는다.
- 폴더마다 임시 영역에 통째로 복사한 뒤 다 복사됐을 때만 제자리로 옮긴다. 가져오기가 중간에 끊기면 다시 실행하면 된다.
- 원본에 올라가지 못한 체크포인트를 가리키는 `ckpt/LATEST.json` 항목과 끊긴 마지막 JSONL 줄은 빼고 가져온다.
  학습 재개와 JSONL 판독기도 이 둘을 건너뛰므로 결과는 같다. 그대로 두면 앞의 것은 그 실행 폴더 전체를, 뒤의 것은 그 파일을
  백업에서 빠지게 한다.
- 가져온 폴더는 백업 커밋에 들어간 뒤에 가져온 것으로 기록한다. 백업이 받지 않는 파일(예: 잘린 JSON)은 빼고 나머지를 백업하며,
  `[storage] imported <폴더>; left out of the backup because incomplete: <파일>`로 알린다. 그 파일은 그 VM에만 남는다.
  폴더의 어떤 파일도 백업에 들어가지 못했다면 `[storage] imported <폴더> is not in the backup yet ...`로 알리고,
  다른 VM에서 다시 가져온다.

## 7. 일반 결과 폴더로 내보내기

공식 라이브러리를 설치한 Mac에서 실행한다. export는 백업을 읽기만 하고 게시하지 않는다.

```bash
python3 drive_storage.py export \
  --auth runs/private/drive-auth.json \
  --out runs/exported-colab-results \
  --skip models --skip cache
```

- 같은 `--out`으로 다시 실행하면 그 폴더를 최신 백업에 맞게 갱신한다. 로컬에서 고친 파일은 덮어쓰지 않고 경고만 남긴다.
- `--commit <ID>`는 합친 최신본 대신 로그에 나온 한 커밋을 내보낸다. 충돌 로그의 다른 버전이 여러 런타임의 전사 캐시를 합친
  것이면 `combined from commits …`로 합친 커밋들을 적으며, 각 커밋을 따로 내보낼 수 있다.
- 이전 코드(`9123e53`)로 만든 내보내기 폴더도 같은 명령으로 갱신한다.
- `--skip <접두어>`는 그 경로를 받지 않으며 여러 번 쓸 수 있다. 그 경로에 전에 받아 둔 파일은 그대로 둔다.
  `models/`(mono 비교군 가중치까지 약 24 GB)와 `cache/`(LibriSpeech 압축 파일 약 6.7 GB)까지 받으면
  전체 내보내기가 매우 커지므로, 결과만 볼 때에는 위처럼 두 경로를 건너뛴다.
- `--skip`은 그 export 실행에만 적용된다. 같은 `--out`으로 `--skip` 없이 다시 실행하면 전에 건너뛴 경로도 받는다.

## 8. 노트북 밖에서 실행한 명령

노트북 세션이 연결되어 있으면 Colab 터미널에서 실행한 명령이 `LOCAL_STORE_ROOT`에 쓴 결과 파일도
120초마다 주기 백업에 들어간다. 바로 백업하거나 `models/`·`cache/`까지 백업하려면 노트북 셀에서 다음을 실행한다.

```python
storage.flush()
```

노트북 세션이 `LOCAL_STORE_ROOT`를 쓰는 동안 `drive_storage.py backup --local /content/vcsasr-store`는
같은 작업 폴더를 두 곳에서 쓰지 않도록 거부된다.

인증한 계정을 바꿀 때에는 기존 파일과 섞지 않도록 새로운 `LOCAL_STORE_ROOT`도 선택한다.
Drive 5TB 용량은 Colab 로컬 디스크 크기와 별개다.
