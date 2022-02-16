# Deep learning demos

In this repository, you will find some scripts used to perform some deep learning demos. I'm using these scripts to run some deeplearning demos on a remote GPU and feed it with images captured with the webcam on my laptop. This is useful for being able to use the power of GPUs for, say, demos during lectures. If you want to do the same, in addition to the client/server provided here, I'm also using ssh tunneling scripts to forward the ports of the server running on the remote GPU on my localhost with [these scripts](https://github.com/jeremyfix/deeplearning-lectures/tree/master/ClusterScripts).

If necessary, the client/server handles JPEG compression/decompression. That might be useful for low bandwidth networks. 

- [General client/server documentation](general-clientserver-documentation)
- [Installation](installation)
- [Semantic segmentation](semantic-segmentation)
	- [Detectron2](detectron2)
	- [Semantic segmentation pytorch (MIT CSAIL)](semantic-segmentation-pytorch-mit-csail)
- [Depth estimation](depth-estimation)
    - [From big to small](from-big-to-small)

## Acknowledgment

These developments have been released within the [FEDER Grone project](https://interreg-grone.eu/)

## General client/server documentation

For the server, an executable entry point dldemos_server is installed in your PATH.

	$ dldemos_server
	usage: dldemos_server [-h] [--port PORT] [--jpeg_quality JPEG_QUALITY]
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
	  --config CONFIG       The config to load If you wish to use aconfig
                            provided by the deeplearning_demos package, use
                            --config config://

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

There is also a generic server server.py, doing useless stuff on an image just to show how to program one.

## Installation

Install [pytorch](https://pytorch.org/get-started/locally/). While writting this guide, pytorch-1.5.1 was used.

We also need python3-opencv :

    apt install python3-opencv

By default, we use it OpenCV for JPEG encoding the image we send over the network but if you want, you can also use TurboJPEG which requires an additional dependency:

    python3 -m pip install --user PyTurboJPEG
    
you may also need to install the libturbojpeg :

    sudo apt install libturbojpeg0-dev
    
Then you can clone clone the repository:

	git clone --recursive https://github.com/jeremyfix/deeplearning_demos.git

In the following, we denote the variable DEEPLEARNING_DEMOS_PATH the path of the deeplearning_demos clone on your drive. For example, if you ran the command above from your home, DEEPLEARNING_DEMOS_PATH=~/deeplearning_demos

## Semantic segmentation

### Detectron2

Here, we use detectron2 from facebook research [github link](https://github.com/facebookresearch/detectron2/). 

Fortunately, detectron2 provides very easy to use installers on Linux and I recommand you to follow their [procedure for
installing from pre-built](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#install-pre-built-detectron2-linux-only). At the time of writting, I succesfully tested with detectron2 v0.6, on torch 1.9 with cuda 11.1.

Once installed, you should be able to:

	python3 -c "import detectron2; from detectron2 import _C"

If that fails, be sure to strictly follow the installation instructions of detectron2.

Then you can start the server with for example :

    dldemos_server --config config://detectron2-panopticseg.yaml

And then the client

    dldemos_client --host localhost --port 6008 

Note that if you run the dldemos_server on a remote host, a simple ssh tunnel to forward the port 6008 on your home
machine and you can then send an image grabbed locally on the remote host for processing and getting back the results.

### Semantic segmentation pytorch (MIT CSAIL)

Here we use the semantic-segmentation-pytorch from the MIT CSAIL [github link](https://github.com/CSAILVision/semantic-segmentation-pytorch). The code is already cloned into the deeplearning_libs subdirectory. Howevern you still need to copy in an `__init__.py` script to be able to import it as a module.

Take the [__init__.py](./share/semantic_segmentation_pytorch__init__.py) script, and copy it in the clone semantic_segmentation_pytorch directory and rename it as `__init__.py` (making a symbolic link to keep it updated does not work):

    cd DEEPLEARNING_DEMOS_PATH/deeplearning_libs/semantic_segmentation_pytorch/
    cp ../../share/semantic_segmentation_pytorch__init__.py __init__.py

Then you need to add a path to your `PYTHONPATH`:

    export PYTHONPATH=$PYTHONPATH:DEEPLEARNING_DEMOS_PATH/deeplearning_libs

You should then be able to do:

    python3 -c "import semantic_segmentation_pytorch"

If the above commands fail, you should probably check your PYTHONPATH, or the init script, or .. let me know in the issue ?

Then you can start the server with for example :

    dldemos_server --config config://segmentation_resnet101_upernet.yaml

And then the client

    dldemos_client --host localhost --port 6008 

## Depth estimation

### From big to small

Here we use the [From big to small: multi-scale local planar guidance depth estimation](https://github.com/cogaplex-bts/bts) code which is already cloned in the deeplearning_libs subdirectory. 

However, you still need to bring in an `__init__.py` script to be able to import the pytorch code. The `__init__.py` is basically empty but you still need it.

    cd DEEPLEARNING_DEMOS_PATH/deeplearning_libs/bts
    ln -s ../../share/bts__init__.py __init__.py


Then you need to add a path to your `PYTHONPATH`:

    export PYTHONPATH=$PYTHONPATH:DEEPLEARNING_DEMOS_PATH/deeplearning_libs


And that's it, you can test it with :

    dldemos_server --config config://bts.yaml 

Check the content of the bts.yaml file to adapt to your needs.

Given the depth image is a one channel image, the client needs to be run with a specific argument 

    dldemos_client --host localhost --port 6008 --depth 1


## Using a virtualenv

Experimented on Ubuntu 18.04, with python 3.6.9. The requirements.txt file expects cuda11.3.  

	virtualenv -p python3 venv
	source venv/bin/activate
	python -m pip install -r requirements.txt
	python -m pip install .

And you can then run `dldemos_server`

## Using a slurm cluster

We provide a sbatch file to be run with sbatch on a cluster handled with slurm :

	sbatch slurm.sbatch

It will handle the creation of the virtualenv, install the libraries and start the dldemos_server on the detectron2 semantic segmentation
