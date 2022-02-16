#!/usr/bin/env python3
#coding: utf-8

# Standard modules
import subprocess
import pathlib
import os
import argparse
import socket
# External modules
import torch
import yaml
# Local modules
import deeplearning_demos
from deeplearning_demos.models.detectron2 import Detectron2
from deeplearning_demos.dlserver import Server

import dmoz



def main():
    parser = argparse.ArgumentParser()

    # Arguments for the server
    parser.add_argument('--dmoz', 
                        default='localhost:10000', 
                        type=str,
                        help="The host:port of the dmoz server to which register")

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

    # Argument for the config file defining the library and model to load
    parser.add_argument('--config',
                        type=str,
                        help='The config to load. If you wish to use a'
                        'config provided by the deeplearning_demos '
                        'package, use --config config://',
                        action='store',
                        required=True)

    args = parser.parse_args()

    # Register to the dmoz server
    dmoz_host, dmoz_port = args.dmoz.split(':')
    dmoz_port = int(dmoz_port)
    dmoz_client = dmoz.client.make(dmoz_host, dmoz_port) 

    device = torch.device('cuda')

    # Loads the provided config
    config_path = args.config
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
            config_path = os.path.join(config_path, args.config[9:])
    config = yaml.safe_load(open(config_path, 'r'))

    # Register the demo
    demoname = args.config.split('/')[-1].split('.')[0]
    # the demoname is built from the config file name removing the extension
    dmoz_client.set(demoname,
                    {
                        'status': 'Init',
                        'user': os.environ['USER'],
                        'jobid': os.environ['SLURM_JOB_ID'],
                        dmoz.protocol.DEMOTYPE_KEY: demoname
                    })

    if config['library'] == 'semantic_segmentation_pytorch':
        model = SemanticSegmentationPytorch(device,
                                            config['library_options'])
    elif config['library'] == 'detectron2':
        model = Detectron2(config['library_options'])
    elif config['library'] == 'bts':
        model = BTS(config['library_options'])


    # Create the server
    srv = Server(args.port,
                 args.jpeg_encoder, 
                 args.jpeg_quality,
                 model)
    # Indicate we will be waiting for a connection
    dmoz_client.change(demoname, {
        'machine': socket.gethostname(), 
        'port': args.port
    })

    dmoz_client.change(demoname, {'status': 'Ready to process'})

    try:
        srv.run()
    except:
        dmoz_client.change(demoname, {'status': 'Errored'})

