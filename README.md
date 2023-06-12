# Dynamic Tiling
## Overview
저전력 디바이스에서 고해상도 이미지의 object detection을 효율적으로 진행 가능하게 하는 Dynamic Tiling을 구현한다.
Dynamic Tiling은 기존의 타일링 기법과는 달리 오브젝트와의 거리를 기반으로 오브젝트의 크기를 예측하여 타일 사이즈를 조절하기 때문에
필요한 수준의 타일링만을 진행하여 더욱 효율적인 object detection을 가능하게 한다.  
2023년 6월 기준 타일 사이즈 측정 파트는 미구현 상태이다.

## 세부 기능
![image](https://github.com/2unju/dynamic_tiling/assets/77797199/a5538129-cb5f-4919-8117-e06a8fa3cd07)

## Requirements
버전 세팅은 evaluation 한정이며, inference의 경우 Micropython을 사용하기 때문에 여타 라이브러리의 버전을 맞출 필요가 없음.
- Python == 3.6
- tensorflow == 2.9.1
- Pillow >= 9.5.0

## Tiling
본 repository는 데모(정성 평가)에 사용된 inference 코드와 정량 평가에 사용된 evaluation 코드를 포함한다.
동일한 조건 하에서 정성 평가와 정량 평가를 진행하기 위해 다양한 크기와 수의 오브젝트를 갖는 WIDER FACE 데이터셋을 사용하며,
따라서 face detection에 특화된 FOMO 모델을 사용한다.

### Demo
```text
inference
    └ inference.py
```
- OpenMV IDE와 Micropython을 기반으로 구현
- OpenMV Cam을 필요로 함 (본 프로젝트에서는 Nicla Vision 사용)
- 주어진 타일 사이즈에 기반하여 각 타일에 대해 개별적으로 object detection을 수행 후
OpenMV IDE의 Frame Buffer에 탐지된 오브젝트의 위치를 표시

### 정량 평가
```text
evaluation
    ├ dataloader.py
    ├ evaludation.py
    ├ read_mat_file.py
    ├ utils.py
    └ data
        ├ valid
        │   └ image01.png
        └ eval_tools
              └ ...
```
- Tensforflow Lite 기반으로 구현
- WIDER FACE의 validation set에 대한 성능 평가 진행
- evaludation.py의 실행으로 평가 진행 가능
