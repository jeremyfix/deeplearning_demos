

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# We need to include the current path .... because
# semantic_segmentation_pytorch is importing lib.nn
# which cannot be found if the __file__ directory is
# not within the path
import sys
sys.path.append(dir_path)

import lib
import models
import data
import config
import utils
