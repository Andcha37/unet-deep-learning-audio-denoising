import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from unet_class import UNet
from dataload import get_audio_loaders

BASE_DATA_PATH = "  "

def train():
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # 맥북 GPU 활용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # aws 서버 활용
    print(f"Using device: {device}")

    # 2. 데이터로더 가져오기 (데이터 로더 로직 모듈로 분리)
    train_loader, val_loader, _ = get_audio_loaders(
        noisy_dir=os.path.join(BASE_DATA_PATH, "magnitude_mixed"),
        clean_dir=os.path.join(BASE_DATA_PATH, "magnitude_clean"),
        batch_size=16
    )

    # 2. 모델 및 옵티마이저
    model = UNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # betas=(0.9, 0.999) 파라미터로 사용 가능

    # 3. 학습 루프
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            noisy, clean = batch['input'].to(device), batch['label'].to(device)
            
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            losses = []
            for batch in val_loader:
                noisy, clean = batch['input'].to(device), batch['label'].to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                losses.append(loss.item())

            val_loss_avg = sum(losses) / len(losses)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss_avg:.4f}")    

    torch.save(model.state_dict(), "unet_model.pth")
    print("학습 완료 및 모델 저장 성공!")

if __name__ == "__main__":
    train()