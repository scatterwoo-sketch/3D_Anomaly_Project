# anomalib 최종 버전: Folder 데이터 로더의 모든 필수 인자를 포함한 최종 코드
import torch
from torchvision.transforms import v2 as T
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

print("--- ✅ Anomalib 최종 호환 스크립트 (Folder 로더 사용) ---")

try:
    # 1. 데이터 모듈 생성
    print("STEP 1: 데이터 모듈 생성 중...")
    augmentations = T.Compose([
        T.Resize((256, 256)),
        T.ToDtype(torch.float32, scale=True)
    ])

    # [수정] 누락되었던 필수 인자인 'name'을 추가합니다.
    datamodule = Folder(
        name="thermal_print",          # This line is the fix.
        root="C:/3D_Anomaly_Project/datasets/thermal_print",
        normal_dir="train/good",
        abnormal_dir="test/bad_heat",
        normal_test_dir="test/good",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
        train_augmentations=augmentations,
        val_augmentations=augmentations,
        test_augmentations=augmentations
    )
    print("✅ STEP 1: 데이터 모듈 생성 완료.")

    # 2. Patchcore 모델 생성 (No changes)
    print("\nSTEP 2: Patchcore 모델 생성 중...")
    model = Patchcore(
        layers=["layer2", "layer3"],
        backbone="wide_resnet50_2",
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9
    )
    print("✅ STEP 2: Patchcore 모델 생성 완료.")

    # 3. Anomalib 엔진 생성 (No changes)
    print("\nSTEP 3: Anomalib Engine 생성 중...")
    engine = Engine(
        logger=False,
        accelerator="auto",
        devices=1,
        default_root_dir="C:/3D_Anomaly_Project/results"
    )
    print("✅ STEP 3: Engine 생성 완료.")

    # 4. 학습 및 테스트 실행
    print("\n--- 🚀 학습 시작 ---")
    engine.fit(model=model, datamodule=datamodule)
    print("--- ✅ 학습 완료 ---")

    print("\n--- 🔬 테스트 시작 ---")
    test_results = engine.test(model=model, datamodule=datamodule)
    print("--- ✅ 테스트 완료 ---")

    print("\n\n--- 📊 최종 성능 결과 ---")
    print(test_results)

except Exception as e:
    print(f"\n❌ 치명적인 오류 발생: {e}")