import sys
import os
from processor.recognition_lstm import REC_Processor


if __name__ == "__main__":
    

    args = ['--config', "./config/ntu120-xsub/train.yaml"]
    
    processor = REC_Processor(sys.argv[1:])
    processor.start()