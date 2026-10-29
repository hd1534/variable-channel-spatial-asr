# eval-dev 데이터셋 출처 및 라이선스 고지

이 디렉터리에 수록된 다채널 오디오는 **LibriSpeech ASR 코퍼스의 파생 저작물**이다. 원저작물이 **CC BY 4.0** 라이선스로 배포되므로 파생 오디오 재배포 시에도 원저작물 출처와 변경 내역을 함께 명시해야 하며, 본 문서는 해당 출처 표기 의무를 이행한다.

## 1. 원저작물 정보

| 항목 | 내용 |
|---|---|
| 데이터셋명 | LibriSpeech ASR corpus (`dev-clean` 분할) |
| 저자 | Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur |
| 공식 배포처 | https://www.openslr.org/12 |
| 라이선스 | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |
| 참고문헌 | V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “LibriSpeech: An ASR Corpus Based on Public Domain Audio Books,” in *Proc. IEEE ICASSP*, 2015, pp. 5206–5210. |

합성에 사용한 원본 아카이브(`dev-clean.tar.gz`)의 SHA-256 해시는 다음과 같다.

```text
76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3
```

## 2. 원본에 가한 변환 내역

원본 단채널 녹음을 그대로 재배포하지 않고 아래 음향 합성 절차를 거쳐 생성한 결과물이다.

1. `dev-clean` 내 화자 40명의 발화를 선택하여 `train / val / eval` 분할 간 화자 ID가 겹치지 않도록 격리 배정했다. 본 디렉터리의 오디오에 포함된 화자는 학습 분할에 등장하지 않는다.
2. 각 발화를 `pyroomacoustics`(v0.8.5) 이미지 소스법(ISM)으로 시뮬레이션한 **4채널 실내 임펄스 응답(RIR)**과 컨볼루션했다(가상 방 36개, 음원 위치 72곳, 실측 `RT60` 범위 0.10~1.28초).
3. 수신 음량만으로 음원 거리를 역산하는 지름길 학습을 막기 위해 세션마다 `−26 ~ −14 dBFS` 범위의 무작위 공통 이득을 음원 거리와 독립적으로 적용했다.
4. 세션 길이를 20.0초로 맞추고 `16 kHz · 4채널 · 16-bit PCM` WAV 형식으로 직렬화했다.

전체 합성 코드는 `simulate.py`와 `make_eval_dev.py`에 공개되어 있으며, 고정 시드 `20261029`로 동일하게 재생성된다.

## 3. 파생 저작물의 라이선스 및 재배포 조건

CC BY 4.0은 동일조건변경허락(ShareAlike) 조항을 포함하지 않으나, **파생 저작물을 재배포할 때 원저작물 출처 표기와 변경 내역 고지 의무는 그대로 유지된다.** 따라서 본 디렉터리의 오디오 파일을 외부로 복사하거나 재배포할 때는 본 `ATTRIBUTION.md` 파일을 반드시 함께 포함해야 한다.

## 4. 비음성 구성 요소

- `ground_truth.jsonl`, `meta.jsonl`, `manifest.json`에 수록된 공간 라벨 및 메타데이터는 본 벤치마크 저장소에서 자체 생성한 산출물이다.
- 실내 임펄스 응답 합성에 사용한 `pyroomacoustics`는 MIT 라이선스로 배포되며, 데이터 생성 시에만 호출되었고 라이브러리 코드는 본 디렉터리에 재배포되지 않는다.
