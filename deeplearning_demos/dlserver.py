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
import deeplearning_demos
from deeplearning_demos import utils
from deeplearning_demos.models.semantic_segmentation_pytorch import SemanticSegmentationPytorch
from deeplearning_demos.models.detectron2 import Detectron2


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
            print("Waiting for a connection on localhost:{}".format(port))
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


def main():
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
                        help='The config to load. If you wish to use a'
                        'config provided by the deeplearning_demos '
                        'package, use --config config://',
                        action='store',
                        required=True)

    args = parser.parse_args()

    device = torch.device('cuda')

    # Loads the provided config
    if(len(args.config) >= 9 and
       args.config[:9] == 'config://'):
        config_path = os.path.join(
            os.path.dirname(deeplearning_demos.__file__),
            './configs')
        if(len(args.config) == 9):
            # Check the available configs
            print("Available configs : ")
            print("\n".join(["- " + c for c in os.listdir(config_path)]))
            return
        else:
            args.config = os.path.join(config_path, args.config[9:])

    print("Loading {}".format(args.config))
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


if __name__ == '__main__':
    main()
