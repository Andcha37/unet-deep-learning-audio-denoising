import sys
import os
import torch
from torch.utils.data import DataLoader

# ✨ 상위 폴더 경로 추가 (모듈 import 에러 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ==========================================
# 🛑 [주의] 본인이 만드신 모듈 이름으로 꼭 수정하세요!
# ==========================================
# 1. 본인의 진짜 Dataset 클래스 불러오기
from dataset_utils.dataset import FullDataset # (예시 이름)

# 2. 본인의 진짜 U-Net 모델 불러오기
from unet_module.unet_class import UNet 

'''
데이터셋 모듈을 통해 불러오는 파일이 정확한 shape으로 불러와지는지 확인하는 코드
'''

def test_full_pipeline():
    print("🚀 [Full Pipeline: 파일 1개 관통 테스트 시작]\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 사용 장치: {device}\n")

    # ==========================================
    # 1. 진짜 Dataset 연결 및 파일 1개만 뽑기 (Subset 마법!)
    # ==========================================
    print("⚙️ 1. 데이터셋 및 데이터로더 연결 중...")
    try:
        # 실제 데이터가 있는 진짜 폴더 경로를 적어주세요!
        CLEAN_DIR = "C:/Users/user/unet-project/audio_for_test/clean_wav" 
        NOISE_DIR = "C:/Users/user/unet-project/audio_for_test/noise_wav"

        # ==========================================
        # 1.1. Dataset에 파일 딱 1개만 '리스트' 형태로 주입!
        # ==========================================
        # 💡 주의: FullDataset 클래스의 __init__이 폴더 경로가 아닌 '파일 리스트'를 
        # 받을 수 있도록 수정되어 있어야 합니다!
        mini_dataset = FullDataset(
            CLEAN_DIR, # 👈 파일 1개짜리 리스트 전달
            NOISE_DIR  # 👈 파일 1개짜리 리스트 전달
        )
        
        # DataLoader에 넣기
        dataloader = DataLoader(mini_dataset, batch_size=1, shuffle=False)
        print("✅ 데이터로더 세팅 완료! (전체 스캔 없이 1개만 주입 완료)\n")
        
    except Exception as e:
        print(f"❌ 데이터로더 연결 실패:\n{e}")
        return

    # ==========================================
    # 2. 모델 세팅
    # ==========================================
    print("⚙️ 2. U-Net 모델 준비 중...")
    try:
        model = UNet().to(device)
        model.eval() # 테스트니까 평가 모드
        print("✅ U-Net 모델 준비 완료!\n")
    except Exception as e:
        print(f"❌ 모델 셋업 실패:\n{e}")
        return

    # ==========================================
    # 3. 파이프라인 관통 (DataLoader -> Model)
    # ==========================================
    print("🔥 3. 파이프라인 관통 테스트 가동!")
    try:
        # 데이터로더에서 그 '1개의 파일'을 뽑아냅니다.
        # 본인의 Dataset이 반환하는 순서대로 변수를 받으세요 (예: noisy_mag, clean_mag, phase 등)
        batch_data = next(iter(dataloader))
        
        # 만약 (noisy_mag, clean_mag) 2개를 반환한다면:
        noisy_input = batch_data["input"].to(device)  # (예시) "noisy"
        clean_target = batch_data["label"].to(device) # (예시) "clean"

        print(f"  - 📥 데이터로더가 뱉어낸 Noisy Input 크기: {noisy_input.shape}")
        print(f"  - 🎯 데이터로더가 뱉어낸 Clean Label 크기: {clean_target.shape}")

        # 모델에 통과시킵니다.
        with torch.no_grad():
            output = model(noisy_input)
            
        print(f"  - 📤 U-Net이 뱉어낸 예측값 Output 크기: {output.shape}")
        
        # 4. 최종 검증
        if output.shape == clean_target.shape:
            print("\n🎉 [전체 파이프라인 완벽 통과!] 🎉")
            print("1. 실제 파일 로드 -> 2. STFT 전처리 -> 3. DataLoader 배치 생성 -> 4. U-Net 노이즈 제거까지")
            print("모든 모듈이 톱니바퀴처럼 완벽하게 맞물려 돌아갔습니다!!")
            print("이제 이 상태 그대로 `train.py`만 만들면 됩니다! 🚀🚀🚀")
        else:
            print(f"\n⚠️ [경고] 예측 크기({output.shape})와 정답 크기({clean_target.shape})가 다릅니다.")

    except Exception as e:
        print(f"\n❌ [충돌 발생!] 데이터가 파이프라인을 통과하는 도중 에러가 났습니다:\n{e}")
        print("💡 팁: 에러 메시지의 맨 마지막 줄을 확인하세요. 차원(Shape) 문제인지 타입(Type) 문제인지 알 수 있습니다.")

if __name__ == "__main__":
    test_full_pipeline()