import sys
import os
from processor.recognition import REC_Processor

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__ == "__main__":
        
    processor = REC_Processor(sys.argv[1:])
    processor.start()