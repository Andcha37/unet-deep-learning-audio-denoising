import sys
import os
import traceback

# ✨ 상위 폴더 경로 추가 (모듈 import 에러 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ==========================================
# 🛑 [주의] 본인의 모듈 이름/경로에 맞게 수정하세요!
# ==========================================
# 질문자님의 data_loader 모듈에서 get_dataloaders 함수를 불러옵니다.
from dataloader_utils.data_loader import get_dataloaders  

'''
테스트 용 데이터를 실제 데이터 폴더 경로 내에 wav파일 여러개 집어넣어야 코드가 터지지 않음.
각 loader에 0개가 되면 안됨.
dataloader로 잘 불러오는지 확인하는 코드
'''

def test_my_get_dataloaders():
    print("🚀 [나의 get_dataloaders 모듈 정밀 테스트 시작]\n")

    # 실제 데이터 폴더 경로
    CLEAN_DIR = "C:/Users/user/unet-project/audio_for_test/clean_wav" 
    NOISE_DIR = "C:/Users/user/unet-project/audio_for_test/noise_wav"

    # ==========================================
    # 1. 내 모듈(get_dataloaders) 실행하기
    # ==========================================
    print("⚙️ 1. get_dataloaders 함수 실행 중...")
    try:
        # 💡 주의: get_dataloaders 함수의 인자(parameter) 이름은 본인이 만드신 대로 맞추세요!
        train_loader, val_loader, test_loader = get_dataloaders(
            clean_dir=CLEAN_DIR, 
            noise_dir=NOISE_DIR, 
            batch_size=2  # 테스트니까 작게 2개만!
        )
        print("✅ 데이터로더 덩어리 생성 성공!\n")
    except Exception as e:
        print("❌ get_dataloaders 실행 중 에러 발생 (경로나 인자 이름을 확인하세요):")
        traceback.print_exc()
        return

    # ==========================================
    # 2. 첫 번째 배치(Batch) 뽑아내기 및 정밀 해부
    # ==========================================
    print("🔥 2. 데이터로더에서 첫 배치를 뽑아 해부합니다...")
    try:
        # 이 순간 FullDataset의 __getitem__ 이 작동하며 STFT 전처리가 돌아갑니다!
        batch_data = next(iter(train_loader))

        print("\n========================================")
        print(f" 📦 뱉어낸 데이터의 자료형: {type(batch_data)}")
        print("========================================")

        # 🔍 딕셔너리(Dict) 형태로 반환하는지 확인
        if isinstance(batch_data, dict):
            print("💡 아하! '딕셔너리' 형태로 반환하고 있습니다. (아까 에러 원인!)")
            print("👇 모델에 넣을 때 아래의 Key(이름)를 사용해야 합니다:")
            for key, value in batch_data.items():
                print(f"  - Key: ['{key}'] ➔ 텐서 크기: {value.shape}")
                
        # 🔍 튜플/리스트 형태로 반환하는지 확인
        elif isinstance(batch_data, (tuple, list)):
            print("💡 '튜플/리스트' 형태로 반환하고 있습니다.")
            for i, value in enumerate(batch_data):
                print(f"  - Index: [{i}] ➔ 텐서 크기: {value.shape}")
        else:
            print(f"⚠️ 예상치 못한 데이터 형태입니다: {type(batch_data)}")

        print("\n🎉 [검사 완료] 모듈들이 완벽하게 연결되어 전처리된 텐서를 뱉어냅니다! 🚀")

    except Exception as e:
        print(f"\n❌ [충돌 발생!] 데이터를 뽑는 과정(__getitem__ 등)에서 에러가 났습니다:")
        traceback.print_exc()  # 에러가 난 진짜 원인과 줄 번호를 보여줍니다!

if __name__ == "__main__":
    test_my_get_dataloaders()