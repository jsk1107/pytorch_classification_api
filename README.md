# Pytorch Classification API

Pytorch를 활용한 Classification API를 제작하였습니다. config파일만 수정하면 간단히 사용할 수 있습니다.
가상환경(conda, venv)에서 사용하는것을 권장합니다. Dockerfile 또한 함께 제공하고 있습니다. 

### 지원 가능한 모델

- EfficientNet (b0 ~ b7)
- ResNext (50, 101)
- ResNet (18, 34, 50, 101, 152)
- Inception-v3
- VGG-19

---

## Install

`pytorch 1.7.1`, `torchvision 0.8.2` 에서 제작되었습니다.

```Shell
pip install -r requirement.txt
```

상세 패키지는 아래와 같습니다. 만약 설치 도중 에러가 나는 모듈은 별도로 설치해야합니다. 

```Shell
pip install torch==1.7.1
pip install torchvision==0.8.2
pip install tensorboardX
pip install numpy
pip install pandas
pip install opencv-python
pip install PyYAML
pip install tqdm
pip install efficientnet_pytorch
```

---
## How to use

### 1. Set folder structure

시작하기에 앞서 수집된 데이터셋을 아래와 같은 스타일로 반드시 변경해야합니다. 
class별로 train, val의 비율을 8:2 분할하여 데이터셋을 구성하기 때문에 Train, Val 폴더를 구분하지 않아도 됩니다.
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


### 2. Modify config.yaml

학습에 필요한 모든 설정값은 `/cfg/config.yaml`에서 수정하여 사용합니다.
변수명을 통해 의미를 파악할 수 있으며, 중요한 내용만 설명하겠습니다

- root_dir(str): 이미지 데이터가 들어있는 최상위 경로를 설정합니다. 데이터 구조를 반드시 지켜야합니다.  
- val_size(float): 검증에 사용될 비율을 설정합니다. 0.2로 설정하게되면 각 클래스마다 20%씩 검증데이터셋으로 분류됩니다.
- resize(list: [int, int]): 이미지의 크기를 조절합니다. Height, Width 순서입니다. 
- model(str): 사용될 모델을 작성합니다. 사용할 수 있는 모델의 종류를 주석처리해놓았습니다. 대소문자를 구분하여 작성해주세요.
- model_path(str): 학습이 진행되면서 Best 모델이 저장될 경로입니다. 변경을 하는것은 권장하지 않습니다.(하단 설명 참조)
- resume(str): 학습된 모델을 불러올때, 모델의 경로를 지정합니다.
- project_name(str): 프로젝트명을 작성합니다.

### 3. Training

학습은 `train.py`에서 시작합니다. IDE 도구를 활용할 경우 `parser.add_argment()` 부분에 있는 config 파일의 경로를 수정하면 됩니다.
commend line을 활용할 경우 아래의 명령어를 통해 실행합니다.

```shell
python train.py -c ${workspace}/cfg/config.yaml

# 학습이 진행되면 아래와 같이 출력됩니다.
Namespace(batch_size=32, config_file='./cfg/config.yaml', cuda=False, epoch=50, gamma=0.9, gpu_ids=0, inference_dir=None, lr=0.001, milestones=[10, 20], model='efficientnet-b0', model_path='./run/test/model_best.pth.tar', num_workers=8, pin_memory=True, project_name='test', resize=[224, 224], resume=None, root_dir='D:/data/img/', start_epoch=0, tensorboard=True, val_size=0.2)
==> Create label_map & path
Done!
Loaded pretrained weights for efficientnet-b0
==> Cuda is not available. CPU Only
Start Epoch: 0
Total Epoch: 50
EPOCH : 0 | Train loss : 0.696:   9%|▉         | 1/11 [00:18<03:07, 18.75s/it]
```

학습이 진행됨과 동시에 몇가지 Log file이 생성됩니다. `/run/`파일이 생성되고 하위에 특정 규칙에 따른 폴더구조가 함께 생성이 됩니다. 

```text
/run/
  | --- {project_name}
  |         |-- {model_name}
  |         |        |-- {TODAY}_{실행횟수}
  |         |        |          |-- tensorboard file 
  |         |        |          |-- checkpoint.pth.tar
  |         |        |          |-- logging.log
  |         |        |          |-- best_pred.txt
  |         |        |-- model_best.pth.tar
 
  
 Ex. Project_name: test / model: efficientnet-b0 / 실행횟수: 2번 
/run/
  | --- test
  |       |-- efficientnet-b0
  |       |        |-- 20201220_0
  |       |        |       |-- events.out.tfevents...
  |       |        |       |-- checkpoint.pth.tar
  |       |        |       |-- logging.log
  |       |        |       |-- best_pred.txt
  |       |        |-- 20201220_1
  |       |        |       |-- events.out.tfevents...
  |       |        |       |-- checkpoint.pth.tar
  |       |        |       |-- logging.log
  |       |        |       |-- best_pred.txt
  |       |        |-- model_best.pth.tar
```

- `config.yaml`에 작성된 project_name, model 문자열을 가진 폴더들이 생성이 됩니다. 그 하위에는 실행한 날짜와 실행횟수를 가진 폴더가 생성됩니다.
중간에 Error가 발생하거나 학습을 강제중지 시킬경우, 다음번 실행이 이전 기록에 덮어쓰기가 되지 않도록 설계하였습니다.
`events.out.tfevents...` 파일은 Tensorboard 파일입니다.
  ```shell
    tensorboard --logdir=./run/{project_name}/{model}/{TODAY}_{CNT}/
    
    ex.
    tensorboard --logdir=./run/test/efficientnet-b0/20201220_1/
    ```

- `checkpoint.pth.tar` 파일은 Epoch마다 덮어쓰기 형태로 저장되며 가장 최근 모델 1개만 존재합니다.

- `logging.log`파일에서는 Epoch별로 CunfusionMatrix을 확인할 수 있습니다.

- `best_pred.txt`는 학습이 진행되면서 가장 성능이 좋은 모델의 F1_Score를 나타냅니다. 

