# Autonomous Slipper

12월 프로젝트 #1

WEBCAM: 1920x1080 (31FPS)

### PLAN

-   #1. Test Autonomous Slipper with Lane Detection

    -   Camera Calibration
    -   Advanced Lane Detection
    -   Lane Detection and Motor Control
    -   Steering DP model

-   #2. Automated control system with Map Localization

    -   Lidar Sensor Test
    -   SLAM ROS
    -   Python Lidar SLAM
    -   Control System with Physics using Lidar
    -   Control System with DP model.

-   #3. Control Slipper Cluster System
    -   ...

## 배운 것들

## Camera Calibration

카메라의 파라미터를 추정하는 과정을 카메라 캘리브레이션이라고 한다.

파라미터를 추정하는 과정이 필요한 이유는 실세계의 3D 점과 캘리브레이션된 카메라로 캡처된 이밎의 해당 2D 투영(픽셀)간의 정확한 관계를 결정하는데 필요하기 때문이다.

복구해야하는 파라메터 종류

-   카메라/렌즈 시스템의 내부 파라미터: 초점거리, 광학 중심, 렌즈의 방사 왜곡 계수
-   외부 파라메터: 이것은 일부 세계 좌표계에 대한 카메라의 방향을 나타낸다.

![효과](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdl3vHY%2FbtrdBqky97A%2FSPAVUg2WkbmXljImqdJAOK%2Fimg.jpg)

카메라 캘리브레이션의 알고리즘

-   입력: 2D 이미지 좌표와 3D 세계 좌표가 알려진 점들을 포함하는 이미지 모음
-   출력: 3x3 카메라 내부 행렬, 각 이미지의 회전 및 이동.

## Reference

-   [https://foss4g.tistory.com/1665](https://foss4g.tistory.com/1665)
