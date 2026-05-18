import torch
import os
import sys

# ✨ 상위 폴더 경로 추가 (모듈 import 에러 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ✨ 본인이 만든 U-Net 클래스를 불러옵니다 (경로는 상황에 맞게 수정)
from unet_module.unet_class import UNet 

def run_pilot_test():
    print("🚀 [U-Net 파일럿 테스트 시작]\n")
    
    # 1. 모델 생성
    try:
        model = UNet()
        print("✅ 1. 모델 클래스 생성 성공!")
    except Exception as e:
        print(f"❌ 모델 생성 중 에러 발생: {e}")
        return

    # 2. 가짜 데이터(Dummy Tensor) 만들기
    # [배치 크기, 채널, 높이, 너비] -> 우리는 [2, 1, 256, 256]을 씁니다.
    # 💡 배치를 1이 아닌 2로 하는 이유: BatchNorm 레이어는 배치가 1이면 에러가 날 수 있기 때문입니다.
    dummy_input = torch.randn(16, 1, 256, 256) 
    print(f"📦 2. 가짜 입력 데이터 준비 완료: {dummy_input.shape}")

    # 3. 모델에 통과시키기 (Forward)
    print("⚙️ 3. 모델에 데이터 통과시키는 중 (Forward)...")
    try:
        # 역전파 계산을 안 하므로 메모리를 아끼기 위해 no_grad() 사용
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"🎉 4. 모델 통과 성공! 출력 데이터 크기: {output.shape}")
        
        # 5. 크기 검증 (제일 중요!)
        if dummy_input.shape == output.shape:
            print("\n✨ [테스트 완벽 통과!] 입력과 출력의 크기가 [16, 1, 256, 256]으로 정확히 일치합니다. 당장 학습을 시작해도 좋습니다! 🚀")
        else:
            print(f"\n⚠️ [경고!] 출력 크기({output.shape})가 입력과 다릅니다. Conv2d의 padding이나 pooling 후 해상도를 다시 확인해 보세요.")

    except Exception as e:
        print(f"\n❌ [에러 발생!] 모델 내부에서 충돌이 일어났습니다:\n{e}")
        print("💡 팁: 에러 메시지를 보면 어느 층(Layer)에서 크기(Shape)가 안 맞아서 터졌는지 알 수 있습니다. 특히 Skip Connection(torch.cat) 부분을 확인하세요!")

if __name__ == "__main__":
    run_pilot_test()