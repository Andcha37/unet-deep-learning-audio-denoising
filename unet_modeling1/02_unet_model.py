import torch
import torch.optim as optim
import os
import sys
from pathlib import Path 

# ✨ 상위 폴더 경로 추가 (모듈 import 에러 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unet_module.unet_class import UNet
from dataloader_utils.data_loader import get_dataloaders

BASE_DIR = Path(os.path.abspath(".."))

def train():
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # 맥북 GPU 활용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # aws 서버 활용
    print(f"Using device: {device}")

    clean_folder = BASE_DIR / "LibriSpeech_Segments"
    noise_folder = BASE_DIR / "UrbanSound8K"

    train_loader, val_loader, _ = get_dataloaders(
        clean_dir = clean_folder,
        noise_dir = noise_folder,
        batch_size=16,
        num_workers=4
    )

    print("데이터셋 준비완료!!")

    print("모델 학습 중....")
    # 2. 모델 및 옵티마이저
    model = UNet().to(device)
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # betas=(0.9, 0.999) 파라미터로 사용 가능

    # 3. 학습 루프
    num_epochs = 1   # 에폭 설정
    best_val_loss = float('inf')  # 최저 검증 로스 기억할 변수

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            noisy, clean = batch['input'].to(device), batch['label'].to(device)
            
            optimizer.zero_grad()
            mask = model(noisy)
            output = mask * noisy
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 10번째 배치마다 현재 로스 출력하기
            if (idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            losses = []
            for batch in val_loader:
                noisy, clean = batch['input'].to(device), batch['label'].to(device)
                mask = model(noisy)
                output = mask * noisy
                loss = criterion(output, clean)
                losses.append(loss.item())

            val_loss_avg = sum(losses) / len(losses)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss_avg:.4f}")  

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg

            torch.save(model.state_dict(), "unet_model.pth")
            print(f"가장 최근 베스트모델 저장 완료 (Best Loss: {best_val_loss})")  

    print("학습 완료 및 모델 저장 성공!")

if __name__ == "__main__":
    train()