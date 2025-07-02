from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# 이미지 크기 및 정규화를 위한 값 (ImageNet 기준)
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_dataloaders(data_dir, batch_size=32, val_split=0.2):
    """
    데이터셋을 위한 DataLoader를 생성합니다.
    학습 데이터에는 증강을, 검증 데이터에는 기본 변환만 적용합니다.
    """
    # 데이터 증강 및 변환 정의
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
    }

    # ImageFolder를 사용하여 전체 데이터셋 로드
    full_dataset = datasets.ImageFolder(os.path.join(data_dir))
    
    # 클래스 이름 저장
    class_names = full_dataset.classes
    
    # 학습/검증 데이터셋으로 분리
    num_train = len(full_dataset)
    val_size = int(num_train * val_split)
    train_size = num_train - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 각 데이터셋에 맞는 변환 적용
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, class_names