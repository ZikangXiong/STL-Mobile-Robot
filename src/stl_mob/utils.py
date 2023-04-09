import os
from os.path import dirname, abspath

import stl_mob

DATA_DIR = os.path.join(dirname(dirname(dirname(abspath(stl_mob.__file__)))), 'data')
PROJ_DIR = dirname(abspath(stl_mob.__file__))
