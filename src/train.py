import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import copy

# 프로젝트 내 다른 모듈 임포트
from model import create_fire_detection_model
from dataset import get_dataloaders

def train_model(data_dir, model_save_dir, num_epochs=25, batch_size=32, learning_rate=0.001):
    """
    모델 학습을 위한 메인 함수

    Args:
        data_dir (str): 데이터셋이 있는 상위 디렉토리 경로.
        model_save_dir (str): 학습된 모델을 저장할 디렉토리 경로.
        num_epochs (int): 총 학습 에포크 수.
        batch_size (int): 배치 크기.
        learning_rate (float): 학습률.
    """
    # 1. 장치 설정 (GPU 우선 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 데이터 로더 준비
    print("Loading data...")
    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size)
    print(f"Classes found: {class_names}")

    # 3. 모델, 손실 함수, 옵티마이저 준비
    print("Initializing model...")
    model = create_fire_detection_model(num_classes=len(class_names))
    model = model.to(device)

    # 손실 함수
    criterion = nn.CrossEntropyLoss()

    # 옵티마이저 (전이 학습이므로, 새로 추가된 레이어의 파라미터만 학습)
    # model.fc의 파라미터만 학습하도록 필터링합니다.
    params_to_update = model.fc.parameters()
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    # 4. 학습 루프
    best_val_loss = float('inf')
    best_model_wts = None

    for epoch in range(num_epochs):
        print("-" * 20)
        print(f"Epoch {epoch+1}/{num_epochs}")

        # 각 에포크는 학습 단계와 검증 단계를 거침
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
                dataloader = train_loader
            else:
                model.eval()   # 모델을 평가 모드로 설정
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            
            # tqdm을 사용하여 데이터 반복
            progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} Phase")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 옵티마이저 그래디언트 초기화
                optimizer.zero_grad()

                # 순전파 (forward)
                # 학습 단계에서만 그래디언트 계산
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우에만 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계 수집
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 검증 단계에서 최고 성능 모델 저장
            if phase == 'val' and epoch_loss < best_val_loss:
                print(f"Validation loss decreased ({best_val_loss:.4f} --> {epoch_loss:.4f}). Saving model...")
                best_val_loss = epoch_loss
                # 모델의 state_dict를 깊은 복사하여 저장
                best_model_wts = copy.deepcopy(model.state_dict())

    # 5. 최고 성능 모델 가중치 저장
    if best_model_wts:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        save_path = os.path.join(model_save_dir, "best_fire_detection_model.pth")
        torch.save(best_model_wts, save_path)
        print(f"Best model saved to {save_path}")
    else:
        print("Training did not improve the model. No model was saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="화재 감지 모델 학습 스크립트")
    
    # 현재 파일 위치를 기준으로 기본 경로 설정
    # train.py는 src/ 안에 있으므로, 상위 디렉토리로 이동해야 data와 models에 접근 가능
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--data-dir", type=str, default=os.path.join(base_dir, "../data/train"),
                        help="학습 데이터가 있는 디렉토리 경로")
    parser.add_argument("--model-save-dir", type=str, default=os.path.join(base_dir, "../models"),
                        help="학습된 모델을 저장할 디렉토리 경로")
    parser.add_argument("--num-epochs", type=int, default=25, help="총 학습 에포크 수")
    parser.add_argument("--batch-size", type=int, default=32, help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="학습률")

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        model_save_dir=args.model_save_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )