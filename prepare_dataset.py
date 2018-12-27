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

from __future__ import absolute_import, division, print_function

import os

from dataset.mnist import mnist_input
from dataset.svhn import svhn_input

if __name__ == "__main__":
    mnist_input.prepare_dataset(os.path.join('./dataset', 'mnist'))
    svhn_input.prepare_dataset(os.path.join('./dataset', 'svhn'))