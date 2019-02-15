import os 
import tensorflow as tf 

from prepare_dataset import init_unified_datasets

class PrepareDatasetTest(tf.test.TestCase):
    
    def testInitUnifiedDatasets(self):
        init_unified_datasets('dataset', 'debug/data')

if __name__ == "__main__":
    tf.test.main()

