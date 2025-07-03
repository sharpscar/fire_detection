import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import copy

#  GPU 전용 강제 설정
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU가 감지되지 않았습니다. 이 코드는 GPU에서만 실행됩니다.")

device = torch.device("cuda")
print(f"GPU 사용 준비 완료: {torch.cuda.get_device_name(device)}")

# 프로젝트 내 다른 모듈 임포트
from model import create_fire_detection_model
from dataset import get_dataloaders

def train_model(data_dir, model_save_dir, num_epochs=25, batch_size=32, learning_rate=0.001):
    """
    GPU 전용: 모델 학습 메인 함수
    """
    # 1. 장치 설정 (GPU만 사용)
    # 전역으로 device가 설정되어 있으므로 여기서는 중복 체크를 제거합니다.
    print(f"Using device: {device}")

    # 2. 데이터 로더 준비
    print("Loading data...")
    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size)
    print(f"Classes found: {class_names}")
    print(f"Train dataset size: {len(train_loader.dataset)} images")
    print(f"Validation dataset size: {len(val_loader.dataset)} images")

    # 3. 모델, 손실 함수, 옵티마이저 준비
    print("Initializing model...")
    model = create_fire_detection_model(num_classes=len(class_names))
    model = model.to(device)

    # 손실 함수 및 옵티마이저는 모델이 device로 이동된 후에 정의합니다.
    criterion = nn.CrossEntropyLoss()
    params_to_update = model.fc.parameters()
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    # 3.1. 저장된 모델 로딩 (있다면)
    model_path = os.path.join(model_save_dir, "best_fire_detection_model.pth")
    if os.path.exists(model_path):
        print(f"저장된 모델에서 재개: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        best_model_wts = copy.deepcopy(model.state_dict())

        print("Performing initial validation pass...")
        model.eval()
        initial_running_loss = 0.0
        initial_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                initial_running_loss += loss.item() * inputs.size(0)
                initial_running_corrects += torch.sum(preds == labels.data)
        best_val_loss = initial_running_loss / len(val_loader.dataset)
        initial_acc = initial_running_corrects.double() / len(val_loader.dataset)
        print(f"Initial Validation Loss: {best_val_loss:.4f} Acc: {initial_acc:.4f}")
        model.train()
    else:
        print("저장된 모델을 찾을 수 없습니다. 새롭게 학습을 시작합니다.")
        best_val_loss = float('inf')
        best_model_wts = None

    # 4. 학습 루프

    for epoch in range(num_epochs):
        print("-" * 30)
        print(f"📅 Epoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else val_loader

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} Phase")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"📊 {phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_loss < best_val_loss:
                print(f" 검증 손실 감소: {best_val_loss:.4f} → {epoch_loss:.4f}, 모델 저장")
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    if best_model_wts:
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(model_save_dir, "best_fire_detection_model.pth")
        torch.save(best_model_wts, save_path)
        print(f"✅ 최종 모델 저장 완료: {save_path}")
    else:
        print("학습 개선 없음. 모델 저장 안 됨.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="화재 감지 모델 학습 (GPU 전용)")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--data-dir", type=str, default=os.path.join(base_dir, "../data/train"),
                        help="학습 데이터 디렉토리")
    parser.add_argument("--model-save-dir", type=str, default=os.path.join(base_dir, "../models"),
                        help="모델 저장 디렉토리")
    parser.add_argument("--num-epochs", type=int, default=25, help="학습 반복 횟수")
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
