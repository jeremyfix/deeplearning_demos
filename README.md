# Deep learning demos

In this repository, you will find some scripts used to perform some deep learning demos.

- [Semantic segmentation](semantic-segmentation)
	- [Installation](installation)
	- [Usage](use)

## General client/server documentation

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

For the client
		
	$ python3 deeplearning_demos/client.py
    usage: client.py [-h] --host HOST --port PORT [--jpeg_quality JPEG_QUALITY]
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

## Semantic segmentation

### Installation 

For the semantic segmentation demos, the script uses :

- detectron2 from facebook research [github link](https://github.com/facebookresearch/detectron2/)
- semantic-segmentation-pytorch from the MIT CSAIL [github link](https://github.com/CSAILVision/semantic-segmentation-pytorch)

You can get this repositories and clone them somewhere in your pythonpath, for example

	mkdir -p ~/GIT/deeplearning_libs
	git clone https://github.com/facebookresearch/detectron2.git ~/GIT/deeplearning_libs/detectron2
	git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git ~/GIT/deeplearning_libs/semantic_segmantation_pytorch
	export PYTHONPATH=$PYTHONPATH:~/GIT/deeplearning_libs

Please note that for the semantic_segmentation_pytorch, the output directory has been renamed with "_" instead of "-" for later being able to import it within python.


For the semantic_segmentation_pytorch lib, since they do not provide any init script, we need to copy one into the directory. Take the [__init__.py](./share/semantic_segmentation_pytorch__init__.py) script, and copy it in the clone semantic_segmentation_pytorch directory and rename it as `__init__.py`.

You should then be able to do:

    python3 -c "import semantic_segmentation_pytorch"
    python3 -c "import detectron2"

If the above commands fail, you should probably check your PYTHONPATH, or the init script, or .. let me know in the issue ?

### Usage

For running the server :

	python3 deeplearning_demos/server.py 

You can now run the client, which is going to grab an image from the webcam, with something like 

	python3 c
