# eval-dev 출처 표기

이 디렉터리의 오디오는 **LibriSpeech의 파생물**이다. LibriSpeech가 CC BY 4.0이므로
출처 표기 의무가 파생물에도 따라온다. 이 파일은 그 의무를 이행한다.

## 원저작물

| | |
|---|---|
| 제목 | LibriSpeech ASR corpus (`dev-clean` 부분집합) |
| 저자 | Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur |
| 출처 | http://www.openslr.org/12/ |
| 라이선스 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 인용 | *LibriSpeech: An ASR Corpus Based on Public Domain Audio Books*, ICASSP 2015 |

내려받은 아카이브 `dev-clean.tar.gz`의 SHA256:

```
76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3
```

## 가한 변경

원본을 그대로 재배포하지 않는다. 아래 변환을 거친 결과물이다.

1. `dev-clean`에서 화자 40명의 발화를 골라 train / val / eval 로 분할한다. 화자가 분할
   경계를 넘지 않으므로 이 디렉터리의 오디오에 쓰인 화자는 학습 분할에 등장하지 않는다.
2. 각 발화를 pyroomacoustics 0.8.5 이미지 소스법으로 생성한 **4채널 실내 임펄스
   응답**과 컨볼루션한다. 방 36개, 음원 위치 72곳, RT60 0.10~1.28초.
3. 세션마다 공통 게인을 −26~−14 dBFS 범위에서 무작위로 적용한다. 거리와 독립적으로
   뽑으므로 수신 레벨에서 거리를 읽을 수 없다.
4. 20초 세션으로 잘라 16 kHz 4채널 int16 PCM WAV로 쓴다.

전체 절차는 `simulate.py`와 `make_eval_dev.py`에 있고, 시드 `20261029`로 비트 단위
재생성된다.

## 이 파생물의 라이선스

CC BY 4.0은 ShareAlike 조항이 없으므로 파생물의 라이선스를 강제하지 않는다. 다만
**원저작물 표기 의무는 이 파생물을 다시 배포할 때도 유지된다.** 이 디렉터리의 오디오를
어디에 옮기든 이 파일을 같이 옮겨라.

## 음성 외 구성요소

`ground_truth.jsonl`, `meta.jsonl`, `manifest.json`의 라벨과 메타데이터는 이
저장소에서 생성한 것이며 LibriSpeech 파생물이 아니다.

RIR 생성에 쓴 pyroomacoustics는 MIT 라이선스다. 생성에만 쓰였고 코드가 재배포되지는
않는다.
