# coding: utf-8
'''
This script provides an interface to the semantic segmentation library
of the MIT CSAIL
'''


# Standard modules
import os
# External modules
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import wget

try:
    from scipy.io import loadmat
    import semantic_segmentation_pytorch
    from semantic_segmentation_pytorch.utils import colorEncode
    import semantic_segmentation_pytorch.lib.utils
except ImportError:
    print("Cannot import semantic_segmentation_pytorch")


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
