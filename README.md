# 교통 표지판 이미지 분류 프로젝트
![image](https://user-images.githubusercontent.com/94281700/169782374-d96e277c-7133-49ca-ac7c-0e66dcbe2b7c.png)

# 개발 환경
### OS : Window 10 pro
### 개발 언어: Python 3.10
### Library & open source
- Tensorflow 2.0
- Keras
- Pandas
- Numpy
- OpenCV
- Matplotlib

# 프로젝트 개요

### 학습 데이터 소개

GTSRB - German Traffic Sign Recognition Benchmark

- Multi-class, single-image classification problem
- 43 classes in total
- More than 50,000 images in total
- Large, lifelike database

### 프로젝트 주제 및 선정 배경
- 교통 표지판은 도로에서 가치있는 정보를 제공
- 도로 교통 표지판은 카메라에 정확하게 인식할 필요가 있음

### 프로젝트 개요
- 교통 표지판 이미지 데이터를 분석하고 딥러닝 모델을 통하여 표지판 종류를 예측하는 multi class classifiaction

### 기대 효과
- 모델이 교통 표지판을 정확하게 인식하고 분류하여 자율 주행 시 교통 표지판 인식에 도움을 준다

# 프로젝트 진행 과정

## 1. 데이터 전처리
- gray scale
- normalization
- data generator

## 2. 학습 모델 파라미터 및 모델 요약
#### 파라미터
- batch_size_val = 50
- steps_per_epoch_val = 100
- epochs_val =10
- imageDimesions = (32, 32, 3)
- testRatio = 0.2
- validationRatio = 0.2

## 3. 모델 정확도
![image](https://user-images.githubusercontent.com/94281700/169780851-db498c39-c773-4204-9df0-a4b32716455d.png)

## 4. 프로젝트 시연 영상
- 유튜브
https://youtu.be/GVTeDUWCv9k

## 5. 결과
- 테스트 데이터 셋 99%의 정확도를 얻음
## 6. 문제점
#### 실제 시연했을시 실제 이미지(=새로운 이미지)에 대해 낮은 정확성을 보임 => 과적합

원인 1. 이미지 증강을 했음에도 데이터가 굉장히 불균형한 상태로 학습

원인 2. 더 다양한 환경을 가진 이미지로 학습이 필요(낮, 밤, 눈, 비, 흐린날 등등)

## 7. 추후 발전 방향
1. 주피터에서 상대적으로 무거운 모델을 수행 후 model.save한 후 그대로 로컬로 가져와 시연 파일에서 실행
(코렙은 opencv 비디오 캡쳐를 사용할 수 없습니다.)

2. 일부 클래스의 데이터 수를 늘리고 다양한 모델 생성 후 비교하여 정확도가 높은 모델로 Test 진행 
