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
    
    config = os.path.join("C:\Personal\Sourecode\school\year3\FYP\GCN-AAE\config\\ntu120-xsub", 'train.yaml')
    # Feeder = import_class("feeders.feeder.Feeder")
    args = ['--config', config]
    
    processor = REC_Processor(args)
    processor.start()