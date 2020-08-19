docker run -it --rm -p 8888:8888 \
	-v /home/jsk/data/cifar:/data/cifar/ \
	-v /home/jsk/workspace/cifar10:/workspace/cifar10 \
	jsk/pytorch:1.4-cuda10.1-cudnn7-runtime \
	jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='mieryu'
