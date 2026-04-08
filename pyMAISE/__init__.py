import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # necessary for tf.probability compatibility with tf-keras

# Determine if display is terminal or notebook
try:
    import IPython

    IS_NOTEBOOK = "Terminal" not in IPython.get_ipython().__class__.__name__

except (NameError, ImportError):
    IS_NOTEBOOK = False

from pyMAISE.postprocessor import PostProcessor
from pyMAISE.settings import ProblemType, init
from pyMAISE.tuner import Tuner
from pyMAISE.utils import Boolean, Choice, Fixed, Float, Int, _try_clear

_try_clear()

# This should always be the last line of this file
__version__ = "2.1.0"
