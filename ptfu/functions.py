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

def splitlist(orig_list, n):
    ''' orig_listで与えられるリスト(tupleでも可)をn分割する。 
    nは1以上、len(orig_list)以下でなくてはならない '''
    splitted = []
    length = len(orig_list)
    len_splitted = length // n
    for i in range(n):
        start = i * len_splitted
        end = start + len_splitted
        if end >= length:
            end = length
        #partial_list = orig_list[start:end]
        #print('len_splitted=', len_splitted, ', len(partial_list)=', len(partial_list))
        splitted.append(orig_list[start:end])
    return splitted

def autodetect_storetype(srclist):
    ''' srclistを与えてStoreTypeを自動判定する
    srclist: アーカイブファイルのパス名のコレクション。 '''
    from ptfu.dataset.storetype import StoreType
    import os.path
    typedict = {}
    for src in srclist:
        stype = StoreType.fromsrcstring(src)
        if stype is not None:
            if stype in typedict:
                typedict[stype] += 1
            else:
                typedict[stype] = 1
    # 一番多いものを正解とする
    max = 0
    maxtype = None
    for key in typedict:
        if typedict[key] > max:
            max = typedict[key]
            maxtype = key
    return maxtype

def random_split_index(minibatchsize, nsplit):
    ''' minibatchsize個のデータをnsplit個のリストから取り出すときの割り振りを行う。
    たとえば、16個のデータを4個のリストから取り出すときに、[8, 2, 0, 6]という個数のリストを生成する。'''
    import random
    generated = []
    residue = minibatchsize
    for i in range(nsplit-1):
        ranint = random.randint(0, residue)
        generated.append(ranint)
        residue -= ranint
    # 最後の1個は帳尻合わせ
    last = residue
    generated.append(last)
    random.shuffle(generated)
    return generated



        
    
