# Copyright 2018 Xu Chen All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np 
import os 

import logging
import daiquiri

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)

def save_to_npz(x, y, data_dir, filename):
    """Save numpy arrays into a npz file
    Args:
        x: images numpy array;
        y: labels numpy array;
        data_dir: directory to store npz file;
        filename: filename of the npz file.
    """
    assert x.shape[0] == y.shape[0]
    np.savez(os.path.join(data_dir, filename), x=x, y=y)
    logger.info("'{}' saved to '{}'".format(filename, os.path.join(data_dir, filename)))
    