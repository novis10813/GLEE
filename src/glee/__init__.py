from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("instructnav-glee-backend")
except PackageNotFoundError:
    __version__ = "0+unknown"

from .config import add_glee_config
from .config_deeplab import add_deeplab_config
# from .GLEE import GLEE
# from .data import build_detection_train_loader, build_detection_test_loader
from .backbone.swin import D2SwinTransformer
from .backbone.eva02 import D2_EVA02

