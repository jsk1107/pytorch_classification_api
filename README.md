# Pytorch Classification API

Pytorch를 이용하여 Classification API를 제작하였습니다. Clone하여 config파일만 수정하면 간단히 사용할 수 있습니다. 또한 Dockerfile을 제공하기 때문에 어디에서나 사용 가능합니다.

#### 지원 가능한 모델
- ResNet(2020.07.23)

## Install

개발환경은 `Ubuntu18.04LTS`, `pytorch 1.4.0`, `torchvision 0.5.0` 입니다.

```Shell
pip install torch==1.4.0
pip install torchvision==0.5.0
pip install tensorboardX
pip install opencv-python
pip install PyYAML
```

설치 도중 에러가 나는 모듈은 별도로 설치해야합니다.

- _`Numpy` 패키지는 1.17를 권장합니다. (2020.07.23 ~)_

## Training

학습은 `main.py`에서 시작합니다. 시작하기에 앞서 수집된 데이터셋을 아래와 같은 스타일중 한가지로 반드시 변경해야합니다.

### 1. Set folder structure

#### Case 1 : 파일 이름으로 Class가 구분된 경우

```
root_dir/
    | --- img/
    |      |-- 1_{class1}.jpg
    |      |-- 2_{class2}.jpg
    |      |-- 3_{class3}.jpg
    |      |-- ...
    |
```

데이터 수집단계에서 위처럼 수집이 된 경우입니다. 소스코드 내부에서 `_`를 기준으로 파싱하여 뒤쪽에 있는 class명을 가져오는 방법으로 데이터를 불러옵니다.

#### Case 2: 폴더명으로 Class가 구분된 경우

```
root_dir/
    | --- cat
    |      |-- 0001.jpg
    |      |-- 0002.jpg
    |      |-- 0003.jpg
    |      |-- ...
    |
    | --- dog
    |      |-- 0001.jpg
    |      |-- 0002.jpg
    |      |-- 0003.jpg
    |      |-- ...
    |
    | --- rabbit
    |      |--...
```
