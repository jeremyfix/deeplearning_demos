#!/usr/bin/env python3
'''
This scripts listens to a port for an incoming image to be processed
It performs the semantic segmentation and returns the segmented image
'''

# Standard modules
import argparse
import os
import socket
import yaml
# External modules
import torch
import torch.nn
from torchvision import transforms
import numpy as np
from scipy.io import loadmat
import wget
from PIL import Image
# Local modules
# import semantic-segmentation-pytorch as semseg
# Delayed because using argparse
import utils

try:
    import semantic_segmentation_pytorch
    from semantic_segmentation_pytorch.utils import colorEncode
    import semantic_segmentation_pytorch.lib.utils
except ImportError:
    print("Cannot import semantic_segmentation_pytorch")

try:
    import detectron2
    from detectron2.engine import DefaultPredictor \
            as detectron2_DefaultPredictor
    from detectron2.config import get_cfg \
            as detectron2_get_cfg
    from detectron2.utils.visualizer import Visualizer \
            as detectron2_Visualizer
    from detectron2.data import MetadataCatalog \
            as detectron2_MetaDataCatalog
except ImportError:
    print("Warning: cannot import detectron2")


class SemanticSegmentationPytorch:

    def __init__(self, device,
                 cfg):
        self.device = device
        semseg_path = os.path.dirname(semantic_segmentation_pytorch.__file__)
        self.colors = loadmat(os.path.join(semseg_path,
                                           'data/',
                                           'color150.mat'))['colors']
        semseg_config_path = os.path.join(semseg_path,
                                          'config',
                                          cfg['model_config'])
        cfg = semantic_segmentation_pytorch.config.cfg
        cfg.merge_from_file(semseg_config_path)

        # Gets the parameters for the encoder/decoder
        arch_encoder = cfg.MODEL.arch_encoder
        arch_decoder = cfg.MODEL.arch_decoder
        fc_dim = cfg.MODEL.fc_dim
        num_class = cfg.DATASET.num_class
        suffix = '_' + cfg.VAL.checkpoint
        ckpt_dir = os.path.join(semseg_path, cfg.DIR)

        # Loads the parameters for preprocessing the images
        self.padding_constant = cfg.DATASET.padding_constant
        self.imgSizes = cfg.DATASET.imgSizes
        self.imgMaxSize = cfg.DATASET.imgMaxSize
        self.normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
        )

        model_name = cfg.DIR.split('/')[-1]

        # Check if the checkpoint dir exists, create otherwise
        if not os.path.exists(ckpt_dir):
            print("Creating {}".format(ckpt_dir))
            os.makedirs(ckpt_dir)

        # Check for the encoder weights
        weights_encoder_path = os.path.join(ckpt_dir,
                                            'encoder' + suffix)
        if not os.path.exists(weights_encoder_path):
            print("Downloading the pretrained encoder weights")
            url = "http://sceneparsing.csail.mit.edu/model/pytorch/" \
                  + model_name\
                  + "/" + 'encoder' + suffix
            print("URL : {}".format(url))
            url = wget.download(url, out=weights_encoder_path)
        # Check for the decoder weights
        weights_decoder_path = os.path.join(ckpt_dir,
                                            'decoder' + suffix)
        if not os.path.exists(weights_decoder_path):
            print("Downloading the pretrained decoder weights")
            url = "http://sceneparsing.csail.mit.edu/model/pytorch/" \
                  + model_name\
                  + "/" + 'decoder' + suffix
            print("URL : {}".format(url))
            url = wget.download(url, out=weights_decoder_path)

        # Network Builders
        builder = semantic_segmentation_pytorch.models.ModelBuilder()
        net_encoder = builder.build_encoder(
            arch=arch_encoder,
            fc_dim=fc_dim,
            weights=weights_encoder_path)

        net_decoder = builder.build_decoder(
            arch=arch_decoder,
            fc_dim=fc_dim,
            num_class=num_class,
            weights=weights_decoder_path,
            use_softmax=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)

        self.segmentation_module = semantic_segmentation_pytorch.models.SegmentationModule(net_encoder,
                                                                                           net_decoder,
                                                                                           crit)

        self.segmentation_module.to(device)
        self.decoder = self.segmentation_module.decoder
        self.encoder = self.segmentation_module.encoder

    def round2nearest_multiple(self, x):
        # Round x to the nearest multiple of p and x' >= x
        return ((x - 1) // self.padding_constant + 1) * self.padding_constant

    def preprocess_img(self, ndimg, imgSize):
        img = Image.fromarray(ndimg, "RGB")
        ori_width, ori_height = img.size

        # calculate target height and width
        scale = min(imgSize / float(min(ori_height, ori_width)),
                    self.imgMaxSize / float(max(ori_height, ori_width)))
        target_height = int(ori_height * scale)
        target_width = int(ori_width * scale)

        # to avoid rounding in network
        target_height = self.round2nearest_multiple(target_height)
        target_width = self.round2nearest_multiple(target_width)

        # resize
        img_resized = img.resize((target_width, target_height), Image.BILINEAR)

        # image to float
        img_resized = np.float32(np.array(img_resized))/255.
        img_resized = img_resized.transpose((2, 0, 1))
        img_resized = self.normalize(torch.from_numpy(img_resized))
        img_resized = torch.unsqueeze(img_resized, 0)

        return img_resized.contiguous()

    def __call__(self, ndimage):
        '''Process an image and returns an image with labels

        Arguments:
            ndimage : ndarray

        Returns:
            ndimage : ndarray of labels
            '''

        # And proceed
        self.segmentation_module.eval()
        with torch.no_grad():
            mean_scores = None
            for imgSize in self.imgSizes:
                torch_img = self.preprocess_img(ndimage, imgSize)
                ori_height, ori_width = ndimage.shape[:2]
                torch_img = torch_img.to(self.device)
                feed_dict = {'img_data': torch_img,
                             'seg_label': None}
                img_scores = self.segmentation_module(feed_dict,
                                                      segSize=(ori_height,
                                                               ori_width))
                if mean_scores is None:
                    mean_scores = img_scores
                else:
                    mean_scores += img_scores
            # above, pred is  (1, 150, height, width)
            # the second value of max is the argmax
            mean_scores *= 1.0 / len(self.imgSizes)
            _, pred_idx = torch.max(mean_scores, dim=1)
        # Post process the result to get the
        # output to be display
        pred_idx = semantic_segmentation_pytorch.lib.utils.as_numpy(pred_idx.squeeze(0).cpu())
        pred_color = colorEncode(pred_idx,
                                 self.colors)
        return pred_color.astype(np.uint8)


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

class Server:

    def __init__(self, port, jpeg_encoder, jpeg_quality, image_processing):
        self.jpeg_handler = utils.make_jpeg_handler(jpeg_encoder, jpeg_quality)
        self.image_processing = image_processing
        self.tmp_buffer = bytearray(7)
        # Creates a temporary buffer which can hold the
        # largest image we can transmit
        self.tmp_view = memoryview(self.tmp_buffer)
        self.img_buffer = bytearray(9999999)
        self.img_view = memoryview(self.img_buffer)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('', port))  # '' means any interface

    def run(self):
        print("The server is ready and listening !")
        self.socket.listen(1)
        while True:
            print("Waiting for a connection")
            conn, addr = self.socket.accept()
            with conn:
                print('Got connection from {}'.format(addr))
                try:
                    while True:
                        utils.recv_data_into(conn, self.tmp_view[:5], 5)
                        cmd = self.tmp_buffer[:5].decode('ascii')
                        if(cmd == 'image'):
                            # Read the image buffer size
                            utils.recv_data_into(conn, self.tmp_view, 7)
                            img_size = int(self.tmp_buffer.decode('ascii'))

                            # Read the buffer content
                            utils.recv_data_into(conn,
                                                 self.img_view[:img_size],
                                                 img_size)
                            # Decode the image
                            raw_img = self.img_view[:img_size]
                            img = self.jpeg_handler.decompress(raw_img)

                            # Process it
                            res = self.image_processing(img)

                            # Encode the image
                            res_buffer = self.jpeg_handler.compress(res)

                            # Make the reply
                            reply = bytes("image{:07}".format(len(res_buffer)),
                                          "ascii")
                            utils.send_data(conn, reply)
                            utils.send_data(conn, res_buffer)
                            utils.send_data(conn, bytes('enod!', 'ascii'))
                        elif cmd == 'quit!':
                            break
                        else:
                            print("Got something else")
                    print("Quitting")
                except RuntimeError as e:
                    print("Error : {}".format(e))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Arguments for the server
    parser.add_argument('--port',
                        default=6008,
                        type=int,
                        help="The port on which to listen"
                             " to an incoming image",
                        action='store'
                        )
    parser.add_argument('--jpeg_quality', type=int,
                        help='The JPEG quality for compressing the reply',
                        default=50)
    parser.add_argument('--jpeg_encoder', type=str, choices=['cv2', 'turbo'],
                        help="Which library to use to encode/decode in JPEG "
                             "the images",
                        default='cv2')

    # Arguments if you want to process a single image
    parser.add_argument('--image',
                        type=str,
                        help='The image to process',
                        action='store',
                        default=None)

    # Argument for the config file defining the library and model to load
    parser.add_argument('--config',
                        type=str,
                        help='The config to load',
                        action='store',
                        required=True)

    args = parser.parse_args()

    device = torch.device('cuda')

    # Loads the provided config
    config = yaml.safe_load(open(args.config, 'r'))

    if config['library'] == 'semantic_segmentation_pytorch':
        model = SemanticSegmentationPytorch(device,
                                            config['library_options'])
    elif config['library'] == 'detectron2':
        model = Detectron2(config['library_options'])

    # Testing on a single image
    if args.image is not None:
        print("Processing the image {}".format(args.image))
        ndimg = np.array(Image.open(args.image).convert('RGB'))
        pred_color = model(ndimg)
        Image.fromarray(pred_color).save("result.png")

    # Builds up the server
    server = Server(args.port,
                    args.jpeg_encoder, args.jpeg_quality,
                    model)

    # Start the server!
    server.run()
