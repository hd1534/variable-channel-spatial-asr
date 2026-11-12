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

연결 셀은 공식 라이브러리를 설치하고, 마지막 완료 백업을 복원한 다음 실제 쓰기를 확인한다.
이 단계가 성공해야 모델 준비·학습을 시작한다. Google API 요청 한도는 개인 프로젝트에도 적용되며,
일시적인 요청 제한, 서버 오류, 네트워크 끊김(DNS 실패, 응답 끊김, 토큰 갱신 중 오류 포함)은 약 3분 동안 지수 백오프로 재시도한다.
용량 부족·권한 취소·API 미활성화는 성공으로 숨기지 않는다.

개인 계정에서 Colab과 Drive를 함께 사용한다면 `STORAGE_BACKEND="mounted_drive"`로
기존 `google.colab.drive.mount()`를 선택할 수 있다. 이 경우 개인 OAuth 설정은 필요 없다.

## 4. 백업 완료와 복원 기준

- `runs/`, 평가 결과, RIR, 전사 캐시는 120초마다 백업한다. 주기 외에 전송 시간도 필요하다.
- 대용량 `models/`, `cache/`는 외부 명령이 끝날 때 백업한다. 그 전에 작은 학습 결과부터 확정한다.
- 전송은 청크 단위로 수행하고, 업로드가 끝나면 Drive의 **크기·MD5**를 확인한다.
- 모든 파일 전송이 성공한 뒤에만 완료 기록을 게시한다. 복원 때 **SHA-256**도 검증한다.
- 명령 종료와 마지막 셀은 백업이 끝날 때까지 기다린다. `[storage] backup complete`를 확인한다.
- 백업 오류는 실행 중인 작업에 전달하며, 로컬 파일을 보존한다. 오류가 해소되면 연결 셀부터 다시 실행한다.

**갑작스러운 Colab VM 종료 전 아직 백업하지 못한 작업은 유실될 수 있다.**
공식 API를 사용해도 아직 전송하지 않은 데이터를 복구할 수는 없다.
네트워크가 잠깐 끊긴 경우 살아 있는 프로세스 안에서 청크 업로드를 재시도한다.
VM이 초기화되어 업로드 세션을 잃으면 마지막 완료 백업을 복원하고 필요한 파일을 다시 전송한다.

같은 백업 폴더는 **한 런타임에서만** 사용한다. 독립된 동시 실행은 별도의 인증 출력 파일과 새 폴더로 설정한다.
서로 다른 런타임에서 이력이 갈리면 자동 덮어쓰기를 중단하고 양쪽 이력을 남긴다.

백업 폴더에는 변경 불가능한 `objects/`와 `commits/`가 보인다. 웹에서 원래 `runs/` 트리가 보이는 형식은 아니다.
동일 내용은 중복 전송하지 않지만, 이전 체크포인트의 원격 이력은 자동 삭제하지 않는다.
따라서 장기간 실험하면 최신 로컬 결과보다 큰 공간을 사용할 수 있다.
학교 Drive의 기존 파일은 자동으로 옮기거나 지우지 않는다.

## 5. 일반 결과 폴더로 내보내기

공식 라이브러리를 설치한 Mac에서 실행한다. 출력은 빈 디렉터리를 지정한다.

```bash
python3 drive_storage.py export \
  --auth runs/private/drive-auth.json \
  --out runs/exported-colab-results
```

노트북 밖에서 Colab 명령을 직접 실행했다면 완료 후 다음으로 백업한다.

```bash
python3 drive_storage.py backup \
  --auth /content/.config/vcsasr/drive-auth.json \
  --local /content/vcsasr-store
```

인증한 계정을 바꿀 때에는 기존 파일과 섞지 않도록 새로운 `LOCAL_STORE_ROOT`도 선택한다.
Drive 5TB 용량은 Colab 로컬 디스크 크기와 별개다.
