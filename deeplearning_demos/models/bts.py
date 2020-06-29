# coding: utf-8
"""
This scripts provides an interface to the bts library
"""

# External modules
import cv2
import torch
from torch.autograd import Variable
import numpy as np

try:
    import bts
    from bts.pytorch.bts import BtsModel
except ImportError:
    print("Warning: cannot import bts")


class Params(object):
    pass

class BTS:

    def __init__(self, cfg):

        params = Params()
        params.mode = 'test'
        params.encoder = cfg['encoder']
        params.bts_size = cfg['bts_size']
        params.max_depth = cfg['max_depth']
        self.focal = None

        self.image_shape = (cfg['library_options']['width'],
                            cfg['library_options']['height'])

        # Download and extract the checkpoint if necessary
        bts_checkpoint_dir = os.path.join(os.path.dirname(bts.__file__),
                                          'models')
        if not os.path.exists(bts_checkpoint_dir):
            bts_checkpoint_dir.mkdir()

        #loading the model
        self.model = BtsModel(params=params)
        self.model = torch.nn.DataParallel(self.model)
        checkpoint = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model'))
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.cuda()

    def set_camera_calibration(self):
        # Intrinsic parameters for your own camera
        # camera_matrix = np.zeros(shape=(3, 3))
        # camera_matrix[0, 0] = 5.4765313594010649e+02
        # camera_matrix[0, 2] = 3.2516069906172453e+02
        # camera_matrix[1, 1] = 5.4801781476172562e+02
        # camera_matrix[1, 2] = 2.4794113960783835e+02
        # camera_matrix[2, 2] = 1
        # dist_coeffs = np.array([ 3.7230261423972011e-02, -1.6171708069773008e-01, -3.5260752900266357e-04, 1.7161234226767313e-04, 1.0192711400840315e-01 ])
        
        #bebop
        camera_matrix = np.zeros(shape=(3, 3))
        camera_matrix[0, 0] = 396.17782
        camera_matrix[0, 2] = 322.453185
        camera_matrix[1, 1] = 403.197845
        camera_matrix[1, 2] = 172.320207
        camera_matrix[2, 2] = 1
        dist_coeffs = np.array([-0.001983, 0.015844, -0.003171, 0.001506, 0.0])

        # Parameters for a model trained on NYU Depth V2
        new_camera_matrix = np.zeros(shape=(3, 3))
        new_camera_matrix[0, 0] = 518.8579
        new_camera_matrix[0, 2] = 320
        new_camera_matrix[1, 1] = 518.8579
        new_camera_matrix[1, 2] = 240
        new_camera_matrix[2, 2] = 1

        self.focal = camera_matrix[0, 0]

        R = np.identity(3, dtype=np.float)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(camera_matrix,
                                                           dist_coeffs, R,
                                                           new_camera_matrix,
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

        depth = np.zeros((ndimage.shape[0], ndimage.shape[1]), dtype=np.uint8)
        depth_01 = depth_est[0].cpu().squeeze() / self.params.max_depth
        depth[32:-1-31, 32:-1-31] = np.uint8(np.round(np.clip(depth_01*255, 0, 255)))

        return depth
