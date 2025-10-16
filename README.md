1. 파일 구조 (압축 파일을 풀어서 다음과 같이 파일을 배치)
C:/
└── 3D_Anomaly_Project/
    ├── datasets/
    │   └── thermal_print/
    │       ├── train/
    │       │   └── good/   (정상 이미지)
    │       └── test/
    │           ├── good/   (테스트 정상 이미지)
    │           └── bad_heat/ (테스트 열 이상 이미지)
    ├── results/    (결과 저장 폴더, 구동 시 자동 생성)
    ├── config.yaml
    └── main.py

2. 파이썬 프로그램은 설치 프로그램 폴더에 있는 
python-3.11.2-amd64.exe로 설치
기존에 파이썬이 있으면 제거 하고 이 프로그램을 설치

3. 파이썬 설치가 완료되면 터미널을 열고 다음의 모듈을 설치 
pip install -q "numpy<2.0"
pip install -q anomalib[full]

4. 모듈 설치까지 완료되면 C:/3D_Anomaly_Project 폴더를
VS 코드로 열어서 main.py를 실행하면 됨

VS 코드는 관리자 권한으로 실행 해야 함.

5. 이미지 데이터는 결과를 보기 위해 임의로 생성한 데이터로
실제 자료가 있으면 해당 폴더에 맞게 데이터를 넣어서 실행 해 보면 됨

