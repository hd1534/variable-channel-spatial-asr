# 본 평가셋(static) 출처 및 라이선스 고지

이 디렉터리에 수록된 다채널 오디오는 **LibriSpeech ASR 코퍼스의 파생 저작물**이다. 원저작물이 **CC BY 4.0** 라이선스로 배포되므로 파생 오디오 재배포 시에도 원저작물 출처와 변경 내역을 함께 명시해야 하며, 본 문서는 해당 출처 표기 의무를 이행한다.

## 1. 원저작물 정보

| 항목 | 내용 |
|---|---|
| 데이터셋명 | LibriSpeech ASR corpus (`test-clean` 분할) |
| 저자 | Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur |
| 공식 배포처 | https://www.openslr.org/12 |
| 라이선스 | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| 참고문헌 | V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “LibriSpeech: An ASR Corpus Based on Public Domain Audio Books,” in *Proc. IEEE ICASSP*, 2015, pp. 5206–5210. |

합성에 사용한 원본 아카이브(`test-clean.tar.gz`)의 SHA-256 해시는 다음과 같다.

```text
39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23
```

## 2. 원본에 가한 변환 내역

원본 단채널 녹음을 그대로 재배포하지 않고 아래 음향 합성 절차를 거쳐 생성한 결과물이다.

1. `test-clean` 화자의 발화를 세션마다 화자당 하나씩 골라 쉼 지점에서 턴으로 나눴다. 발화 내용은 자르거나 바꾸지 않았다. 이 화자들은 학습·검증에 쓰지 않는다.
2. 각 턴을 `pyroomacoustics` 이미지 소스법(ISM)으로 시뮬레이션한 **4채널 실내 임펄스 응답(RIR)**과 컨볼루션했다(가상 방 200개, 세션마다 새 방, 실측 `RT60` 범위 0.11~1.32초).
3. 화자는 첫 위치에 머물고 잡음은 더하지 않았다(주 조건과 같은 세션, 같은 음성, 같은 방).
4. 세션마다 무작위 공통 이득과 채널별 미세 이득 차를 적용하고, 세션 길이를 20.0초로 맞춰 `16 kHz · 4채널 · 16-bit PCM` WAV 형식으로 직렬화했다.

전체 합성 코드는 `make_eval_t2.py`와 `training/scenes.py`에 공개되어 있으며, 고정 시드 `20261031`로 다시 만들 수 있다.

## 3. 파생 저작물의 라이선스 및 재배포 조건

CC BY 4.0은 동일조건변경허락(ShareAlike) 조항을 포함하지 않으나, **파생 저작물을 재배포할 때 원저작물 출처 표기와 변경 내역 고지 의무는 그대로 유지된다.** 따라서 본 디렉터리의 오디오 파일을 외부로 복사하거나 재배포할 때는 본 `ATTRIBUTION.md` 파일을 반드시 함께 포함해야 한다.

## 4. 비음성 구성 요소

- `ground_truth.jsonl`, `meta.jsonl`, `manifest.json`에 수록된 라벨 및 메타데이터는 본 벤치마크 저장소에서 자체 생성한 산출물이다. 참조 전사는 LibriSpeech 원문 전사를 그대로 옮긴 것이다.
- 실내 임펄스 응답 합성에 사용한 `pyroomacoustics`는 MIT 라이선스로 배포되며, 데이터 생성 시에만 호출되었고 라이브러리 코드는 본 디렉터리에 재배포되지 않는다.
