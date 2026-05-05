import torch
import os
from unet_class import UNet
from dataload import get_audio_loaders
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

BASE_DATA_PATH = " "  # 데이터 경로

class evaluator:
    def __init__(self, sample_rate=16000, device='cpu'):
        self.sr = sample_rate
        self.device = device

        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.sr, mode='wb').to(device)
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.sr).to(device)
        self.si_sdr = ScaleInvariantSignalDistortionRatio().to(device)

    def compute(self, preds: torch.Tensor, targets: torch.Tensor):
        if preds.ndim == 3:
            preds = preds.squeeze()
        if targets.ndim == 3:
            targets = targets.squeeze()
        
        pesq_val = self.pesq(preds, targets)
        stoi_val = self.stoi(preds, targets)
        si_sdr_val = self.si_sdr(preds, targets)

        return {
            "PESQ": round(pesq_val.item(), 4),
            "STOI": round(stoi_val.item(), 4),
            "SI-SDR": round(si_sdr_val.item(), 4)
        }


def test(batch_size = 1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"현재 사용 장치: {device}")

    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_model.pth", map_location=device))
    model.eval()

    _, _, test_loader = get_audio_loaders(
        noisy_dir = os.path.join(BASE_DATA_PATH, "magnitude_mixed"),
        clean_dir = os.path.join(BASE_DATA_PATH, "magnitude_clean"),
        batch_size = batch_size
    )

    audio_eval = evaluator(sample_rate=16000, device=device)
    total_metrics = {"PESQ": 0.0, "STOI": 0.0, "SI-SDR": 0.0}
    data_len = len(test_loader)


    with torch.no_grad():
        for batch in test_loader:
            noisy, clean = batch['input'].to(device), batch['label'].to(device)

            prediction = model(noisy)

            batch_metrics = audio_eval.compute(prediction, clean)

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
            


