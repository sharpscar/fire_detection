# 화재 감지 클라이언트 (Fire Detection Client)

## 1. 프로젝트 개요

이 프로젝트는 PyTorch를 사용하여 이미지 속의 화재 여부를 감지하는 딥러닝 모델을 학습하고, 학습된 모델을 사용하여 새로운 이미지를 예측하는 클라이언트입니다.
사전 학습된 ResNet18 모델에 전이 학습(Transfer Learning)을 적용하여 효율적으로 화재와 비화재 이미지를 분류합니다.

## 2. 디렉토리 구조

```
fire-detection-client/
├── data/                     # 학습 및 검증용 이미지 데이터
│   ├── train/
│   │   ├── fire/             # (필수) 화재 이미지
│   │   └── non_fire/         # (필수) 비화재 이미지
│   └── val/
│       ├── fire/             # (선택) 검증용 화재 이미지
│       └── non_fire/         # (선택) 검증용 비화재 이미지
├── models/                   # 학습된 모델 가중치(.pth)가 저장되는 곳
├── src/                      # 소스 코드
│   ├── dataset.py          # 데이터셋 로딩 및 전처리
│   ├── model.py            # 신경망 모델 정의
│   ├── train.py            # 모델 학습 스크립트
│   └── predict.py          # 학습된 모델로 예측하는 클라이언트
├── requirements.txt        # 파이썬 의존성 라이브러리 목록
└── README.txt              # 프로젝트 설명서 (현재 파일)
```

## 3. 사용 방법

### 3.1. 환경 설정

먼저, 프로젝트에 필요한 파이썬 라이브러리를 설치해야 합니다.
프로젝트의 최상위 디렉토리(`fire-detection-client`)에서 아래 명령어를 실행하세요.

```bash
pip install -r requirements.txt
```
*참고: PyTorch는 시스템 환경(CUDA 버전 등)에 따라 설치 방법이 다를 수 있습니다. 만약 위 명령어로 설치가 원활하지 않다면, [공식 PyTorch 웹사이트](https://pytorch.org/get-started/locally/)를 참고하여 환경에 맞는 설치 명령어를 사용하세요.*

### 3.2. 데이터 준비

모델을 학습시키기 위해 이미지 데이터를 준비해야 합니다.

1.  **학습 데이터:**
    *   화재 이미지들을 `data/train/fire/` 디렉토리 안에 넣어주세요.
    *   화재가 아닌 이미지들을 `data/train/non_fire/` 디렉토리 안에 넣어주세요.
2.  **검증 데이터 (선택 사항):**
    *   더 정확한 성능 평가를 위해, 학습에 사용되지 않은 별도의 검증용 이미지를 `data/val/fire/` 와 `data/val/non_fire/` 에 각각 넣어주는 것을 권장합니다.
    *   만약 검증 데이터셋을 따로 준비하지 않으면, 학습 데이터의 일부(기본 20%)가 자동으로 검증용으로 사용됩니다.

### 3.3. 모델 학습

데이터 준비가 완료되면, 모델 학습을 시작할 수 있습니다.
`src` 디렉토리로 이동하여 `train.py` 스크립트를 실행하세요.

```bash
cd src
python train.py
```

학습이 진행되면서 각 에포크(Epoch)의 손실(Loss)과 정확도(Accuracy)가 출력됩니다. 학습이 완료되면 검증 데이터셋에서 가장 성능이 좋았던 시점의 모델 가중치가 `models/best_fire_detection_model.pth` 파일로 자동 저장됩니다.

**학습 옵션 변경:**
학습률, 에포크 수, 배치 사이즈 등을 변경하고 싶다면 명령어 뒤에 인자를 추가할 수 있습니다.

```bash
# 에포크 50, 배치 사이즈 64로 학습
python train.py --num-epochs 50 --batch-size 64
```

### 3.4. 화재 감지 예측 실행

학습이 완료되어 `best_fire_detection_model.pth` 파일이 생성되었다면, 이제 새로운 이미지로 화재 여부를 예측할 수 있습니다.
`src` 디렉토리에서 `predict.py` 스크립트를 실행하고, 인자로 분석하고 싶은 이미지의 경로를 전달하세요.

```bash
# src 디렉토리에서 실행
python predict.py <분석할_이미지_파일_경로>

# 예시
python predict.py ../sample_images/fire_example.jpg
python predict.py /Users/scar/Desktop/my_picture.png
```

스크립트를 실행하면, 해당 이미지가 'FIRE'인지 'NON_FIRE'인지에 대한 예측 결과와 신뢰도(%)가 터미널에 출력됩니다.

## 4. 코드 설명

*   **`src/model.py`**: 사전 학습된 ResNet18 모델을 불러와 마지막 분류 레이어만 화재/비화재(2개 클래스) 분류에 맞게 교체합니다.
*   **`src/dataset.py`**: `data` 폴더의 이미지를 PyTorch가 처리할 수 있는 형태로 변환합니다. 학습 시에는 이미지 뒤집기, 자르기, 색상 변경 등 데이터 증강(Data Augmentation) 기술을 적용하여 모델의 일반화 성능을 높입니다.
*   **`src/train.py`**: `dataset.py`로 데이터를 불러오고 `model.py`로 모델을 생성하여 학습 과정을 총괄합니다. 학습 중 검증 손실이 가장 낮아지는 순간의 모델을 `models/` 폴더에 저장합니다.
*   **`src/predict.py`**: 저장된 모델(`best_fire_detection_model.pth`)을 불러와 한 장의 이미지에 대한 예측을 수행하는 클라이언트 스크립트입니다.
