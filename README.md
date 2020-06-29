# Deep learning demos

In this repository, you will find some scripts used to perform some deep learning demos. I'm using these scripts to run some deeplearning demos on a remote GPU and feed it with images captured with the webcam on my laptop. This is useful for being able to use the power of GPUs for, say, demos during lectures. If you want to do the same, in addition to the client/server provided here, I'm also using ssh tunneling scripts to forward the ports of the server running on the remote GPU on my localhost with [these scripts](https://github.com/jeremyfix/deeplearning-lectures/tree/master/ClusterScripts).

If necessary, the client/server handles JPEG compression/decompression. That might be useful for low bandwidth networks. 

- [General client/server documentation](general-clientserver-documentation)
- [Semantic segmentation](semantic-segmentation)
	- [Installation](installation)
	- [Usage](use)

## Acknowledgment

These developments have been released within the [FEDER Grone project](https://interreg-grone.eu/)

## General client/server documentation

For the segmentation server, a binary segmentation_server is installed in your PATH.

	$ segmentation_server
	usage: segmentation_server [-h] [--port PORT] [--jpeg_quality JPEG_QUALITY]
	                           [--jpeg_encoder {cv2,turbo}] [--image IMAGE]
	                           --config CONFIG
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --port PORT           The port on which to listen to an incoming image
	  --jpeg_quality JPEG_QUALITY
	                        The JPEG quality for compressing the reply
	  --jpeg_encoder {cv2,turbo}
	                        Which library to use to encode/decode in JPEG the
	                        images
	  --image IMAGE         The image to process
	  --config CONFIG       The config to load

For the server

	$ python3 deeplearning_demos/server.py
	usage: server.py [-h] --port PORT [--jpeg_quality JPEG_QUALITY]
					 [--encoder {cv2,turbo}]

	optional arguments:
	  -h, --help            show this help message and exit
	  --port PORT           The port on which to listen for incoming connections
	  --jpeg_quality JPEG_QUALITY
							The JPEG quality for compressing the reply
	  --encoder {cv2,turbo}
							Which library to use to encode/decode in JPEG the
							images

For the client, you can also use the installed entry point :
		
	$ segmentation_client
    usage: segmentation_client [-h] --host HOST --port PORT [--jpeg_quality JPEG_QUALITY]
                               [--resize RESIZE] [--encoder {cv2,turbo}] [--image IMAGE]
                               [--device DEVICE]
    
    optional arguments:
      -h, --help            show this help message and exit
      --host HOST           The IP of the echo server
      --port PORT           The port on which the server is listening
      --jpeg_quality JPEG_QUALITY
                            The JPEG quality for compressing the reply
      --resize RESIZE       Resize factor of the image
      --encoder {cv2,turbo}
                            Library to use to encode/decode in JPEG the images
      --image IMAGE         Image file to be processed
      --device DEVICE       The id of the camera device 0, 1, ..


## Installation

Install [pytorch](https://pytorch.org/get-started/locally/). While writting this guide, pytorch-1.5.1 was used.

Then you can clone clone the repository:

	git clone --recursive https://github.com/jeremyfix/deeplearning_demos.git

In the following, we denote the variable DEEPLEARNING_DEMOS_PATH the path of the deeplearning_demos clone on your drive. For example, if you ran the command above from your home, DEEPLEARNING_DEMOS_PATH=~/deeplearning_demos

## Semantic segmentation

### Detectron2

Here, we use detectron2 from facebook research [github link](https://github.com/facebookresearch/detectron2/). That repository is cloned in the deeplearning_libs subdir. So

	cd deeplearning_libs

and then follow the installation instructions on the [detectron2 website](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
	
Once installed, you should be able to:

	python3 -c "import detectron2; from detectron2 import _C"

If that fails, be sure to strictly follow the installation instructions of detectron2.

### Semantic segmetnation pytorch (MIT CSAIL)

Here we use the semantic-segmentation-pytorch from the MIT CSAIL [github link](https://github.com/CSAILVision/semantic-segmentation-pytorch). The code is already cloned into the deeplearning_libs subdirectory. Howevern you still need to copy in an `__init__.py` script to be able to import it as a module.

Take the [__init__.py](./share/semantic_segmentation_pytorch__init__.py) script, and copy it in the clone semantic_segmentation_pytorch directory and rename it as `__init__.py`.

	cp share/semantic_segmentation_pytorch__init__.py deeplearning_libs/semantic_segmentation_pytorch/__init__.py

You should then be able to do:

    python3 -c "import semantic_segmentation_pytorch"

If the above commands fail, you should probably check your PYTHONPATH, or the init script, or .. let me know in the issue ?

## Depth estimation

### From big to small

Here we use the [From big to small: multi-scale local planar guidance depth estimation](https://github.com/cogaplex-bts/bts) code which is already cloned in the deeplearning_libs subdirectory. 

However, you still need to bring in an `__init__.py` script to be able to import the pytorch code.

**WIP**



## Usage

For running the server, you basically need to specify a config file. The config file tells which library to use (semantic-segmentation-pytorch or detectron2) and then some specific configurations for these libs. Example config files are provided in the [deeplearning_demos/configs](deeplearning_demos/configs) directory.

For example, for running the detectron2 with panoptic segmentation :

	python3 deeplearning_demos/segmentation_server.py --config ./deeplearning_demos/configs/detectron2-panopticseg.yaml

It will take care of downloading the pretrained weights if required (it does so for semantic-segmentation-pytorch; for detectron2, the libs already takes care of it) and run the server. By default, the server will be running on port 6008. 

You can now run the client, which is going to grab an image from the webcam, send it for processing by the server, and display the returned result. 

	python3 deeplearning_demos/client.py --host localhost --port 6008 

You can pass in additional options, check the help message.
