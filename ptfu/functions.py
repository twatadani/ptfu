''' functions.py: ptfuにおけるユーティリティー関数群 '''

import os
import pickle

def cpu_count():
    ''' このプロセスで使用可能なCPUコア数を返す '''
    cpu_count = 1
    if hasattr(os, 'sched_getaffinity'):
        cpu_count = len(os.sched_getaffinity(0))
    else:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
    return cpu_count

def picklable(obj):
    ''' objがpickle化可能かどうかを返す '''
    try:
        pkl = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if pkl is not None:
            return True
    except:
        return False
    return False

        
    
