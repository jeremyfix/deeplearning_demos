# coding: utf-8
"""
This scripts provides an interface to the bts library
"""

# Standard modules
import os
import pathlib
import zipfile
# External modules
import cv2
import torch
from torch.autograd import Variable
import numpy as np
import wget

try:
    import bts
    from bts.pytorch.bts import BtsModel
except ImportError:
    print("Warning: cannot import bts")


class Params(object):
    pass

class BTS:

    def __init__(self, cfg):

        self.params = Params()
        self.params.mode = 'test'
        self.params.encoder = cfg['encoder']
        self.params.bts_size = cfg['bts_size']
        self.params.max_depth = cfg['max_depth']
        self.params.dataset = cfg['dataset']
        self.params.camera_matrix = cfg['camera_matrix']
        if 'distortion' in cfg:
            self.params.distortion = cfg['distortion']
        else:
            self.params.distortion = None
        self.focal = None

        self.image_shape = (cfg['image_shape']['width'],
                            cfg['image_shape']['height'])

        self.set_camera_calibration()

        # Download and extract the checkpoint if necessary
        bts_checkpoint_dir = os.path.join(os.path.dirname(bts.__file__),
                                          'pytorch',
                                          'models')
        if not os.path.exists(bts_checkpoint_dir):
            print("The checkpoint dir {} does not exist, I create it".format(bts_checkpoint_dir))
            pathlib.Path(bts_checkpoint_dir).mkdir()
        bts_model_checkpoint_dir = os.path.join(bts_checkpoint_dir,
                                                cfg['checkpoint'])
        if not os.path.exists(bts_model_checkpoint_dir):
            print("The checkpoint {} is not available, I download it".format(bts_model_checkpoint_dir))
            base_url = "https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/"
            checkpoint_url = "{}{}.zip".format(base_url, cfg['checkpoint'])
            url = wget.download(checkpoint_url, out=bts_checkpoint_dir)
            archive = zipfile.ZipFile("{}.zip".format(bts_model_checkpoint_dir))
            archive.extractall(path=bts_checkpoint_dir)

        checkpoint_model_file = os.path.join(bts_model_checkpoint_dir, "model")

        #loading the model
        self.model = BtsModel(params=self.params)
        self.model = torch.nn.DataParallel(self.model)
        checkpoint = torch.load(checkpoint_model_file)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.cuda()

    def set_camera_calibration(self):
        # Intrinsic parameters for your own camera
        camera_matrix = np.array(self.params.camera_matrix).reshape((3, 3))
        if self.params.distortion:
            dist_coeffs = np.array(self.params.distortion)
        else:
            dist_coeffs = None

        self.focal = camera_matrix[0, 0]

        R = np.identity(3, dtype=np.float)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(camera_matrix,
                                                           dist_coeffs, R,
                                                           camera_matrix,
                                                           self.image_shape,
                                                           cv2.CV_32FC1)

    def __call__(self, ndimage):
        '''
        Process an image through the model
        and postprocess its output
        Returns a Grayscale image in np.uint8 where 255 = arg.max_depth
        '''
        frame_ud = cv2.remap(ndimage, self.map1, self.map2,
                             interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame_ud, cv2.COLOR_BGR2RGB)
        input_image = frame.astype(np.float32)

        # Normalize the image
        input_image[:, :, 0] = (input_image[:, :, 0] - 123.68) * 0.017
        input_image[:, :, 1] = (input_image[:, :, 1] - 116.78) * 0.017
        input_image[:, :, 2] = (input_image[:, :, 2] - 103.94) * 0.017

        input_image_cropped = input_image[32:-1 - 31, 32:-1 - 31, :]

        input_images = np.expand_dims(input_image_cropped, axis=0)
        input_images = np.transpose(input_images, (0, 3, 1, 2))

        with torch.no_grad():
            image = Variable(torch.from_numpy(input_images)).cuda()
            focal = Variable(torch.tensor([self.focal])).cuda()
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = self.model(image, focal)

        depth = np.zeros((ndimage.shape[0], ndimage.shape[1]), dtype=np.uint16)
        depth_01 = depth_est[0].cpu().squeeze() / self.params.max_depth
        depth[32:-1-31, 32:-1-31] = np.uint16(np.round(np.clip(depth_01*255, 0, 255)))

        return depth
