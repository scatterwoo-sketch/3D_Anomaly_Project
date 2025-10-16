# 필요한 라이브러리 추가 임포트
import os
import torch
from pathlib import Path
from torchvision.transforms import v2 as T
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pytorch_lightning import seed_everything


def main():
    """Anomalib 학습 및 테스트를 위한 메인 함수"""
    print("--- ✅ Anomalib 개선된 스크립트 ---")

    # 1. 설정값들을 스크립트 상단 변수로 분리 (하드코딩 방지)
    # ----------------------------------------------------------------
    # 경로 설정 (pathlib.Path를 사용하여 OS 호환성 확보)
    ROOT_PATH = Path("C:/3D_Anomaly_Project")
    DATASET_PATH = ROOT_PATH / "datasets/thermal_print"
    RESULTS_PATH = ROOT_PATH / "results"

    # 데이터 설정
    IMAGE_SIZE = (256, 256)
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 32
    NUM_WORKERS = 4

    # 모델 설정
    LAYERS = ["layer2", "layer3"]
    BACKBONE = "wide_resnet50_2"
    CORESET_SAMPLING_RATIO = 0.1
    NUM_NEIGHBORS = 9
    
    # 재현성을 위한 시드 고정
    seed_everything(42, workers=True)
    # ----------------------------------------------------------------

    try:
        # 2. 데이터 증강 전략 세분화
        print("STEP 1: 데이터 모듈 생성 중...")
        train_augmentations = T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize(IMAGE_SIZE),
            T.ToDtype(torch.float32, scale=True)
        ])
        
        val_test_augmentations = T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ToDtype(torch.float32, scale=True)
        ])

        datamodule = Folder(
            name="thermal_print",
            root=DATASET_PATH,
            normal_dir="train/good",
            abnormal_dir="test/bad_heat",
            normal_test_dir="test/good",
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_augmentations=train_augmentations,
            val_augmentations=val_test_augmentations,
            test_augmentations=val_test_augmentations
        )
        print("✅ STEP 1: 데이터 모듈 생성 완료.")

        # 2. Patchcore 모델 생성
        print("\nSTEP 2: Patchcore 모델 생성 중...")
        model = Patchcore(
            layers=LAYERS,
            backbone=BACKBONE,
            pre_trained=True,
            coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
            num_neighbors=NUM_NEIGHBORS
        )
        print("✅ STEP 2: Patchcore 모델 생성 완료.")

        # 3. Anomalib 엔진 생성
        print("\nSTEP 3: Anomalib Engine 생성 중...")
        engine = Engine(
            logger=True, 
            accelerator="auto",
            devices=1,
            default_root_dir=RESULTS_PATH
        )
        print("✅ STEP 3: Engine 생성 완료.")

        # 4. 학습 및 테스트 실행
        print("\n--- 🚀 학습 시작 ---")
        engine.fit(model=model, datamodule=datamodule)
        print("--- ✅ 학습 완료 ---")

        print("\n--- 🔬 테스트 시작 ---")
        test_results = engine.test(model=model, datamodule=datamodule)
        print("--- ✅ 테스트 완료 ---")

        # [수정된 부분] 최종 성능 결과 출력
        print("\n\n--- 📊 최종 성능 결과 ---")
        
        if test_results:
            image_auroc = test_results[0].get('image_AUROC')
            image_F1Score = test_results[0].get('image_F1Score')
            print(f"Image-level AUROC: {image_auroc:.4f}")
            print(f"image_F1Score AUROC: {image_F1Score :.4f}")

            if isinstance(image_auroc, float):
                print(f"Image-level AUROC: {image_auroc:.4f}")
            else:
                print(f"Image-level AUROC: {image_auroc or 'N/A'}")

            if isinstance(image_F1Score , float):
                print(f"Pixel-level AUROC: {image_F1Score :.4f}")
            else:
                print(f"Pixel-level AUROC: {image_F1Score or 'N/A'}")
        else:
            print("테스트 결과를 가져오는 데 실패했습니다.")

    except Exception as e:
        print(f"\n❌ 치명적인 오류 발생: {e}")

if __name__ == "__main__":
    main()