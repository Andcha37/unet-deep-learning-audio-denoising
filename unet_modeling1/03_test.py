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
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from postprocess_utils.audio_postprocess import inverse_stft

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

class evaluator:
    def __init__(self, sample_rate=16000, device='cpu'):
        self.sr = sample_rate
        self.device = device

        # 메인 음성 평가 지표 2개만 유지
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.sr, mode='wb').to(device)
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.sr).to(device)

    def compute(self, preds: torch.Tensor, targets: torch.Tensor):
        # inverse_stft 결과물인 2차원 [1, Time_samples]에서 
        # 0번째 차원(Batch)만 정확하게 squeeze(0) 해줍니다. -> [Time_samples] 1차원 벡터 변환
        if preds.ndim == 2 and preds.size(0) == 1:
            preds = preds.squeeze(0)
        if targets.ndim == 2 and targets.size(0) == 1:
            targets = targets.squeeze(0)
        
        pesq_val = self.pesq(preds, targets)
        stoi_val = self.stoi(preds, targets)

        return {
            "PESQ": round(pesq_val.item(), 4),
            "STOI": round(stoi_val.item(), 4)
        }


def test(batch_size = 1):
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"현재 사용 장치: {device}")

    model_path = BASE_DIR / "unet_model.pth"
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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

    audio_eval = evaluator(sample_rate=16000, device=device)
    total_metrics = {"PESQ": 0.0, "STOI": 0.0}
    data_len = len(test_loader)

    print("테스트 시작...")
    with torch.no_grad():
        for batch in test_loader:
            # 1. 스펙트로그램 데이터 로드 ([1, 1, 256, 256] 구조 가정)
            noisy_mag = batch['input'].to(device)
            clean_mag = batch['label'].to(device)
            
            # 복원을 위한 위상(Phase) 데이터 꺼내기 (3차원 [1, 256, 256] 가정)
            noisy_phase = batch['phase'].to(device) 

            # 2. U-Net 예측 ([1, 1, 256, 256])
            prediction_mag = model(noisy_mag)

            # 3. 4차원 모델 출력을 질문자님의 inverse_stft 3차원 스펙에 맞추기 위해 squeeze(1) 처리
            # [1, 1, 256, 256] -> [1, 256, 256]
            pred_mag_3d = prediction_mag.squeeze(1)
            clean_mag_3d = clean_mag.squeeze(1)

            # inverse_stft 함수를 사용한 1D Waveform 음원 복원
            # 출력 형태: [1, Time_samples] 2차원 텐서
            pred_waveform = inverse_stft(pred_mag_3d, noisy_phase)
            clean_waveform = inverse_stft(clean_mag_3d, noisy_phase)

            # 5. 오차 측정 및 점수 환산
            batch_metrics = audio_eval.compute(pred_waveform, clean_waveform)

            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
            
    print("=" * 50)
    print("Final Test Results")
    print("=" * 50)
    for key, value in total_metrics.items():
        score = value / data_len
        print(f"{key} : {score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    test(batch_size = 1)