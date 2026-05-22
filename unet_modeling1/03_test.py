import torch
import os
import sys
import random
import numpy as np
from pathlib import Path

# ✨ 상위 폴더 경로 추가 (모듈 import 에러 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unet_module.unet_class import UNet
from dataloader_utils.data_loader import get_dataloaders

BASE_DIR = Path(os.path.abspath(".."))

# ================================================================
# 테스트 환경에서의 재현성을 위한 글로벌 시드 고정
# ================================================================
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(batch_size = 1):
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"현재 사용 장치: {device}")

    model_path = BASE_DIR / "modeling1_best_unet_model.pth"
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    criterion = torch.nn.L1Loss()

    model.eval()

    clean_folder = BASE_DIR / "LibriSpeech"
    noise_folder = BASE_DIR / "UrbanSound8K"

    # 데이터로더 정의 (주의: batch에 원래 noisy 음원의 phase 정보가 포함되어 있어야 합니다!)
    _, _, test_loader = get_dataloaders(
        clean_dir = clean_folder,
        noise_dir = noise_folder,
        batch_size=batch_size,
        num_workers=4
    )

    total_metrics = {"L1": 0.0}
    data_len = len(test_loader)

    print("테스트 시작...")
    with torch.no_grad():
        for batch in test_loader:
            # 스펙트로그램 데이터 로드 ([1, 1, 256, 256] 구조 가정)
            noisy_mag = batch['input'].to(device)
            clean_mag = batch['label'].to(device)
            
            # U-Net 예측 ([1, 1, 256, 256])
            mask = model(noisy_mag)
            prediction_mag = mask * noisy_mag

            loss = criterion(prediction_mag, clean_mag)
            total_metrics["L1"] += loss.item()
            
    print("=" * 50)
    print("Final Test Results")
    print("=" * 50)
    for key, value in total_metrics.items():
        score = value / data_len
        print(f"{key} : {score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    test(batch_size = 1)