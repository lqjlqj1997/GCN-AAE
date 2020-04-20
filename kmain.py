import sys
import os
from processor.Kmeans import Cluster


if __name__ == "__main__":
    

    args = ['--config', "./config/ntu120-xsub/train.yaml"]
    
    processor = Cluster(sys.argv[1:])
    processor.start()