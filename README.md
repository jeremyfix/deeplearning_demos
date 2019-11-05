# Deep learning demos

In this repository, you will find some scripts used to perform some deep learning demos.


## Semantic segmentation

For the semantic segmentation demos, the script uses :

- detectron2 from facebook research [github link](https://github.com/facebookresearch/detectron2/)
- semantic-segmentation-pytorch from the MIT CSAIL [github link](https://github.com/CSAILVision/semantic-segmentation-pytorch)

You can get this repositories and clone them somewhere in your pythonpath, for example

	mkdir -p ~/GIT/deeplearning_libs
	git clone https://github.com/facebookresearch/detectron2.git ~/GIT/deeplearning_libs/detectron2
	git clone https://github.com/CSAILVision/semantic-segmentation-pytorch.git ~/GIT/deeplearning_libs/semantic_segmantation_pytorch
	export PYTHONPATH=$PYTHONPATH:~/GIT/deeplearning_libs

Please note that for the semantic_segmentation_pytorch, the output directory has been renamed with "_" instead of "-" for later being able to import it within python.


For the semantic_segmentation_pytorch lib, since they do not provide any init script, we need to copy one into the directory. Take the [./share/semantic_segmentation_pytorch__init__.py](__init__.py) script, and copy it in the clone semantic_segmentation_pytorch directory and rename it as `__init__.py`.

You should then be able to do:

    python3 -c "import semantic_segmentation_pytorch"
    python3 -c "import detectron2"

If the above commands fail, you should probably check your PYTHONPATH, or the init script, or .. let me know in the issue ?


