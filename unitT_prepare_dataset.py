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
"""Tests for prepare_dataset"""

import os 
import tensorflow as tf 

from prepare_dataset import init_unified_datasets

class PrepareDatasetTest(tf.test.TestCase):
    
    def testInitUnifiedDatasets(self):
        init_unified_datasets('dataset', 'debug/data')

if __name__ == "__main__":
    tf.test.main()
