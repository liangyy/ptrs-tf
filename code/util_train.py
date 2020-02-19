import util_misc
import re
import tensorflow as tf

def get_phase(phase_yaml):
    if phase_yaml is None:
        phase = {
            'phase1': {
                'optimizer': 'Adam'
                'epoch': 20
            }    
        }
    else:
        phase = util_misc.load_ordered_yaml(phase_yaml)
    
    return phase

# to load spatial information
def get_tss(start, end, strand):
    if strand == '+':
        return start
    else:
        return end
def chr2num(chrm):
    if 'X' in chrm:
        chrm = 23
    elif 'Y' in chrm:
        chrm = 24
    elif 'M' in chrm:
        chrm = 25
    else:
        chrm = int(re.sub('chr', '', chrm))
    return chrm

def str2optimizer(mystr):
    return getattr(tf.keras.optimizers, mystr)()
