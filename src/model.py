import torch
import torch.nn as nn
from torchvision import models

def create_fire_detection_model(num_classes=2, pretrained=True):
    """
    사전 학습된 ResNet18 모델을 기반으로 화재 감지 모델을 생성합니다.

    Args:
        num_classes (int): 분류할 클래스의 수 (화재/비화재 이므로 2).
        pretrained (bool): ImageNet으로 사전 학습된 가중치를 사용할지 여부.
    
    Returns:
        torch.nn.Module: 화재 감지 모델
    """
    # 사전 학습된 ResNet18 모델 로드
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    # 기존 레이어의 가중치는 고정 (fine-tuning 시에는 일부만 고정할 수도 있음)
    for param in model.parameters():
        param.requires_grad = False

    # 마지막 분류 레이어(fc)를 우리의 작업에 맞게 새로운 레이어로 교체
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model

if __name__ == '__main__':
    # 모델이 잘 생성되는지 테스트
    model = create_fire_detection_model()
    print(model)
    # 임의의 이미지 텐서로 테스트
    test_tensor = torch.randn(1, 3, 224, 224)
    output = model(test_tensor)
    print(f"Output shape: {output.shape}") # 예상 출력: [1, 2]