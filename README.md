# Deep learning demos

In this repository, you will find some scripts used to perform some deep learning demos. I'm using these scripts to run some deeplearning demos on a remote GPU and feed it with images captured with the webcam on my laptop. This is useful for being able to use the power of GPUs for, say, demos during lectures. If you want to do the same, in addition to the client/server provided here, I'm also using ssh tunneling scripts to forward the ports of the server running on the remote GPU on my localhost with [these scripts](https://github.com/jeremyfix/deeplearning-lectures/tree/master/ClusterScripts).

If necessary, the client/server handles JPEG compression/decompression. That might be useful for low bandwidth networks. 

- [General client/server documentation](#general-clientserver-documentation)
- [Installation](#installation)
- [Available demos](#available-demos)
- [Using a slurm cluster](#using-a-slurm-cluster)

## Acknowledgment

An early version of these developments have been released within the [FEDER Grone project](https://interreg-grone.eu/)

## General client/server documentation

For the server, an executable entry point dldemos_server is installed in your PATH.

	$ dlserver --help

	usage: dlserver [-h] [--verbose {20,10}] [--port PORT] [--config CONFIG]

	options:
	  -h, --help         show this help message and exit
	  --verbose {20,10}  Verbosity level, INFO(20), DEBUG(10)
	  --port PORT        The port on which to listen to an incoming image
	  --config CONFIG    The config to load. If you wish to use aconfig provided by the deeplearning_demos package, use --config config://

For the client, you can also use the installed entry point :
		
	$ dlclient_cli --help

	usage: dlclient_cli [-h] [--hostname HOSTNAME] [--port PORT] [--device_id DEVICE_ID] [--resize_factor RESIZE_FACTOR]

	options:
	  -h, --help            show this help message and exit
	  --hostname HOSTNAME   The host to connect to
	  --port PORT           The port on which to connect
	  --device_id DEVICE_ID
							The device id to be used for providing the camera input for opencv
	  --resize_factor RESIZE_FACTOR
							The resize factor applied to the grabbed camera before sending

## Installation

For installing the server, you can either clone the repository or install it directly with pip 

```
python3 -m pip install git+https://github.com/jeremyfix/deeplearning_demos.git#subdirectory=dlserver
```

For installing the client, you can proceed the same way :
```
python3 -m pip install git+https://github.com/jeremyfix/deeplearning_demos.git#subdirectory=dlclient
```

## Available demos

The available demos are provided by a yaml file. The default yaml is provided in `dlserver/configs/default.yaml`. At the time of writting this documentation, it features :

- image classification with MobileNet, trained on ImageNet, as provided by [https://github.com/onnx/models](https://github.com/onnx/models)
- image classification with Resnet50, trained on ImageNet, as provided by [https://github.com/onnx/models](https://github.com/onnx/models)
- Object detection with [ultralytics Yolov8n](https://docs.ultralytics.com/tasks/detect/), trained on Coco
- Object detection + Segmentation with [ultralytics Yolov8n](https://docs.ultralytics.com/tasks/segment/), trained on Coco
- Text translation English to French with [t5 base from Hugging Face](https://huggingface.co/t5-base)
- Text translation French to English with [t5 base from Hugging Face](https://huggingface.co/t5-base)

## Using a slurm cluster

We provide a sbatch file to be run with sbatch on a cluster handled with slurm :

	sbatch slurm.sbatch

It will handle the creation of the virtualenv, install the libraries and start the dlserver.
