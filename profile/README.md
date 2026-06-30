# RV LAB

RV LAB은 경기대학교 전자공학부의 Robotics and Vision Lab입니다. 자율주행, 로봇 비전, SLAM, 센서 융합, 휴머노이드/Physical AI를 중심으로 실제 환경에서 동작하는 로봇 시스템을 연구합니다.

이 GitHub 조직은 연구 과정에서 사용하는 코드, 실험 도구, 문서, 데이터셋 관련 자료를 정리하기 위한 공간입니다.

## Research

### Autonomous Driving and SLAM

RV LAB은 카메라, 4D Radar, LiDAR, IMU, RTK-GPS를 함께 사용하는 자율주행 인지와 위치 추정 기술을 연구합니다.

- 4D Radar-Camera 기반 Vector Map SLAM
- Dynamic Object Removal Masks(DORM)를 이용한 Visual-4D Radar Odometry
- IPM과 vanishing point correction을 이용한 vector map generation
- 4D Radar Z projection image와 vector map을 이용한 loop detection
- LiDAR/IMU/RTK-GPS 기반 outdoor SLAM
- 2D LiDAR/IMU 기반 indoor SLAM
- Road marking segmentation과 semantic road mapping

### Sensor Fusion in Real Environments

연구 대상은 실험실 조건에만 머무르지 않습니다. 눈, 비, 안개, 야간 환경처럼 센서가 불안정해지는 상황에서 로봇과 차량이 주변을 인식하고 위치를 추정하는 방법을 다룹니다.

- Camera-4D Radar sensor fusion
- LiDAR-camera calibration and fusion
- Camera/LiDAR desnowing and deraining
- Foggy environment vehicle pose estimation
- ROS 기반 자동차 데이터셋 구축

### Robotics and Physical AI

로봇 분야에서는 휴머노이드, 사족보행 로봇, 매니퓰레이터, rover 플랫폼을 다룹니다. 시뮬레이션에서 학습한 정책을 실제 하드웨어로 옮기는 문제와, 로봇이 실내외 환경을 이해하고 이동하는 문제에 관심이 있습니다.

- Humanoid robotics and biped walking
- Isaac Sim 기반 digital twin simulation
- VR teleoperation and ACT policy learning
- Vision-Language-Action(VLA) and Vision-Language Navigation(VLN)
- Quadruped robot localization for pipe inspection
- Visual SLAM for rover mapping
- Manipulator-based 2D LiDAR 3D measurement

### Vehicle Control and Safety

자율주행 시스템의 하위 제어와 차량 안정성도 주요 연구 주제입니다.

- Anti-lock Braking System(ABS)
- Electronic Stability Control(ESC)
- Side slip angle estimation
- Sliding mode control
- Collision avoidance
- Leader-follower vehicle formation
- Unmanned snow plow control

## Selected Work

- **MSC-RAD4R**: ROS-based automotive dataset with 4D radar, stereo camera, LiDAR, RTK-GPS, GPS, IMU, and wheel data  
  <https://mscrad4r.github.io/home/>
- **4D Radar-Camera Vector Map SLAM**: visual odometry, dynamic object removal, vector mapping, and loop closing for autonomous driving
- **Autonomous Driving Golf Cart**: HD map generation, RTK-GPS localization, LiDAR tracking, and 4D Radar-Camera vehicle pose estimation
- **Camera-LiDAR Fusion for Desnowing**: sensor noise removal and SLAM for snow/rain environments
- **Visual Inspection Quadruped Robot**: localization for pipe inspection using VINS-Fusion maps and 3D CAD models

## Links

- Lab website: <https://sites.google.com/view/rv-lab>
- Research statements: <https://sites.google.com/view/rv-lab/research-statements>
- Publications: <https://sites.google.com/view/rv-lab/publications>
- Projects: <https://sites.google.com/view/rv-lab/projects>

## Affiliation

Department of Electronic Engineering, Kyonggi University
