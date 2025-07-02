import torch
from PIL import Image
import argparse
from model import create_fire_detection_model
from dataset import MEAN, STD, IMG_SIZE
import torchvision.transforms as transforms

def predict(image_path, model_path, device):
    """
    단일 이미지에 대해 화재 여부를 예측합니다.

    Args:
        image_path (str): 예측할 이미지 파일 경로.
        model_path (str): 학습된 모델 가중치 파일 경로.
        device (torch.device): 연산을 수행할 장치 (cpu 또는 cuda).
    """
    # 1. 모델 불러오기
    model = create_fire_detection_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 추론 모드로 설정 (매우 중요!)

    # 2. 이미지 불러오기 및 변환
    # 검증 데이터셋과 동일한 변환을 적용해야 함
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) # 배치 차원 추가

    # 3. 예측 수행
    with torch.no_grad(): # 그래디언트 계산 비활성화
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    # 4. 결과 해석 및 출력
    class_names = ['fire', 'non_fire'] # data/train 폴더 순서에 따라 결정됨
    prediction = class_names[predicted_idx.item()]
    
    print(f"결과: '{prediction.upper()}'")
    print(f"신뢰도: {confidence.item() * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="화재 감지 클라이언트")
    parser.add_argument("image", help="분석할 이미지 파일 경로")
    parser.add_argument("--model", default="../models/best_fire_detection_model.pth", help="학습된 모델 파일 경로")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    predict(args.image, args.model, device)