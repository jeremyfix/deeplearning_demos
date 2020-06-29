# coding: utf-8
"""
This scripts provides an interface to the detectron2 library 
"""

# Standard modules
import os

# try:
import detectron2
from detectron2.engine import DefaultPredictor \
        as detectron2_DefaultPredictor
from detectron2.config import get_cfg \
        as detectron2_get_cfg
from detectron2.utils.visualizer import Visualizer \
        as detectron2_Visualizer
from detectron2.data import MetadataCatalog \
        as detectron2_MetaDataCatalog
# except ImportError:
    # print("Warning: cannot import detectron2")


class Detectron2:

    def __init__(self,
                 cfg):
        self.cfg = detectron2_get_cfg()
        cfg_file = os.path.join(os.path.dirname(detectron2.__file__),
                                cfg['cfg_file'])
        self.output_postprocessing = cfg['output']
        self.cfg.merge_from_file(cfg_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
        self.cfg.MODEL.WEIGHTS = cfg['weights']
        self.predictor = detectron2_DefaultPredictor(self.cfg)

    def __call__(self, ndimage):
        '''
        Process an image through the model
        and postprocess its output
        Returns a RGB image in np.uint8
        '''
        outputs = self.predictor(ndimage[:, :, ::-1])
        v = detectron2_Visualizer(ndimage,
                                  detectron2_MetaDataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        if self.output_postprocessing == 'instance':
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        elif self.output_postprocessing == 'panoptic':
            panoptic_seg, segments_info = outputs["panoptic_seg"]
            v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        return v.get_image()

