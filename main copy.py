from graph.ntu_rgb_d import Graph
import torch
import numpy as np
import random
from util import get_parser
from util import import_class
import argparse

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def main():

    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)

    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()
