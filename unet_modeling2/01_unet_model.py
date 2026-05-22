import torch
import torch.optim as optim
import os
import sys
from pathlib import Path 
import random
import numpy as np

# added libraries
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_ 
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

# ================================================================
# 글로벌 난수 시드 고정
# ================================================================
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # CUDA 환경에서 결정론적 알고리즘 사용하도록 환경변수 설정
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) # 필요 시 주석 해제하여 높은 재현성 적용 가능

set_seed(42)
# =================================================================

# ✨ 상위 폴더 경로 추가 (모듈 import 에러 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unet_module.unet_class import UNet
from dataloader_utils.data_loader import get_dataloaders
from postprocess_utils.audio_postprocess import inverse_stft

BASE_DIR = Path(os.path.abspath(".."))

# ================================================================
# 파이토치 DataLoader 용 멀티워커 시드 초기화 함수
# ================================================================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ================================================================
# 추가됨 : PESQ 평가용 함수 (val 전용) , cpu 사용
# ================================================================
class AudioEvaluator:
    def __init__(self, sample_rate=16000, device='cpu'):
        self.device = device
        self.pesq = PerceptualEvaluationSpeechQuality(fs=sample_rate, mode='wb').to(device)

    def compute(self, pred_waveform, target_waveform):
        self.pesq.reset()

        pred_waveform = torch.clamp(pred_waveform, -1.0, 1.0)
        target_waveform = torch.clamp(target_waveform, -1.0, 1.0)

        try:
            pesq_score = self.pesq(pred_waveform.to(self.device), target_waveform.to(self.device)).item()
        except:
            pesq_score = None
        
        return pesq_score




def train():
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # 맥북 GPU 활용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # aws 서버 활용
    print(f"Using device: {device}")

    clean_folder = BASE_DIR / "LibriSpeech_Segments"
    noise_folder = BASE_DIR / "UrbanSound8K"

    g = torch.Generator()
    g.manual_seed(42)

    train_loader, val_loader, _ = get_dataloaders(
        clean_dir = clean_folder,
        noise_dir = noise_folder,
        batch_size=16,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )

    print("데이터셋 준비완료!!")

    print("모델 학습 중....")
    # 2. 모델 및 옵티마이저
    model = UNet().to(device)
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # betas=(0.9, 0.999) 파라미터로 사용 가능
    
    # ================================================================
    # 추가됨 : 스케쥴러 : T_max 는 에폭에 따라 설정 , 50이면 50에폭때 1e-6
    # ================================================================
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.000001)
    
    # ================================================================
    # 추가됨 : gradscaler : 
    # ================================================================ 
    scaler = torch.amp.GradScaler('cuda')

    # ================================================================
    # 추가됨 : val 데이터셋 전용 evaluator
    # ================================================================
    evaluator = AudioEvaluator(sample_rate=16000, device='cpu')

    # ================================================================
    # 추가됨 : 체크포인트 파일
    # ================================================================
    start_epoch = 0
    
    # 비율 기반 Best Model 저장을 위한 변수
    best_score = -float('inf') 
    
    checkpoint_path = BASE_DIR / "modeling2_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print("기존 체크포인트 모델 발견하여 이어서 학습 진행 중...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        # 이전 체크포인트에 best_score가 없으면 0.0으로 초기화
        best_score = checkpoint.get('best_score', -float('inf'))
        
        print(f"Resume Epoch : {start_epoch}")

    # 3. 학습 루프
    num_epochs = 50   # 에폭 설정

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            noisy, clean = batch['input'].to(device), batch['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True)


            # ====================================================
            # 추가됨 : AMP Forward
            # ====================================================
            with torch.amp.autocast('cuda'):
                pred_mask = model(noisy)
                output = pred_mask * noisy
                loss = criterion(output, clean)

            # ====================================================
            # 추가죔 : AMP Backward
            # ====================================================
            scaler.scale(loss).backward()

            # ====================================================
            # 추가됨 : Gradient Clipping
            # ====================================================
            scaler.unscale_(optimizer)

            clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            # 10번째 배치마다 현재 로스 출력하기
            if (idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        

        model.eval()
        val_losses = []
        total_pesq = 0.0
        valid_pesq_count = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy, clean = batch['input'].to(device), batch['label'].to(device)
                noisy_phase = batch['noisy_phase'].to(device)
                clean_phase = batch['clean_phase'].to(device)

                # ================================================
                # AMP Validation
                # ================================================
                with torch.amp.autocast('cuda'):
                    pred_mask = model(noisy)
                    output = pred_mask * noisy
                    loss = criterion(output, clean)
                val_losses.append(loss.item())

                # ================================================
                # Spectrogram -> Waveform (평가 지표 계산용)
                # ================================================
                # 주의: 학습에는 절대 쓰이지 않고, 오디오 복원용으로만 phase 사용
                output_mag_squeeze = output.squeeze(1).float()
                clean_mag_squeeze = clean.squeeze(1).float()

                pred_waveform = inverse_stft(output_mag_squeeze, noisy_phase.float())
                clean_waveform = inverse_stft(clean_mag_squeeze, clean_phase.float())

                # ================================================
                # Audio Quality Metrics
                # ================================================
                for i in range(pred_waveform.size(0)):
                    metrics = evaluator.compute(pred_waveform[i], clean_waveform[i])
                    if metrics is not None:
                        total_pesq += metrics
                        valid_pesq_count += 1 

        avg_val_l1 = sum(val_losses) / len(val_losses)
        avg_pesq = total_pesq / valid_pesq_count if valid_pesq_count > 0 else 0.0

        # ========================================================
        # Scheduler Step
        # ========================================================
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ========================================================
        # Epoch Log
        # ========================================================
        print("=" * 60)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train L1 Loss: {total_loss/len(train_loader):.4f}")
        print(f"Val L1 Loss    : {avg_val_l1:.4f}")
        print(f"Validation PESQ: {avg_pesq:.4f}")
        print(f"Learning Rate  : {current_lr:.8f}")
        print("=" * 60)

        # ========================================================
        # Save Best Model
        # ========================================================
        current_score = avg_pesq - avg_val_l1 

        if current_score > best_score:
            best_score = current_score
            best_path = BASE_DIR / "modeling2_best_unet_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"최고 성능 모델 갱신 및 저장 완료 (Score: {best_score:.4f} | PESQ: {avg_pesq:.4f}, L1: {avg_val_l1:.4f})")

        # ========================================================
        # Save Full Checkpoint
        # ========================================================
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_score': best_score # best_score 저장으로 변경
        }

        torch.save(checkpoint, checkpoint_path)

        print("체크포인트 저장 완료")

    print("학습 완료 및 모델 저장 성공!")

if __name__ == "__main__":
    train()