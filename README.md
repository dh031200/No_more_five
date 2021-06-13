# No_more_five
5인 이상 집합한 경우를 탐지하는 object detection 프로젝트

### Team 4인이하 집합가능 

#### Member 김동훈 정창수 성재빈 박진영

------------

테이블과 사람을 탐지 후 테이블 갯수에 따라 그룹을 생성합니다.

탐지된 사람은 유클리드 거리 기준 가장 가까운 테이블 그룹에 속하게 됩니다.

그룹의 인원이 5명 이상이 되면 화면의 경계선에 빨간색과 노란색 테두리가 생겨 경고합니다.

![warning_2](https://user-images.githubusercontent.com/66017052/121816067-6d029f00-ccb4-11eb-87b8-c1fca523ef60.gif)

구글 이미지 검색에서 식탁과 사람 각 600장의 사진을 모아 학습시켰습니다.

데이터의 양이 적어 탐지율이 낮지만, 이는 데이터 양을 늘려 충분히 학습시킨다면 더 좋은 결과를 보일 것입니다.

사전에 학습된 가중치를 사용하기 위해 > [model](https://drive.google.com/file/d/1ldfN0nnbZModFHBR1fx2oqR-GG6pgfjQ/view?usp=sharing "학습된 모델") < 다운로드


------------

### 기본 세팅 값
#### 임계값
threshold = 0.7   
#### 입력 사이즈 (yolov4 기본 값: 416)
input_size = 416
#### 감지할 화면 좌측 상단 좌표의 x
left_top_x = 200
#### 감지할 화면 좌측 상단 좌표의 y
left_top_y = 200
#### 감지할 화면 너비
detect_width = 1280
#### 감지할 화면 높이
detect_height = 720
#### 모델 경로
model_path = 'models/'
#### 클래스 정보 및 모델 정보
class_info = 'obj.names'
model = 'yolov4-0613'  # 사용할 모델 'yolov4-first', 'yolov4-0613'
