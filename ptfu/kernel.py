''' kernel.py: ptfuのシステム全体に関連するカーネルを記述する '''

import tensorflow as tf

class Kernel:
    ''' ptfuシステムのカーネル '''

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
        from multiprocessing import Manager
        from .functions import cpu_count
        import random
        ncpu = max(cpu_count() - 1, 1)
        self.texecutor = ThreadPoolExecutor(max_workers=ncpu)
        self.pexecutor = ProcessPoolExecutor(max_workers=ncpu)
        self.manager = Manager()
        self.training_tensor = tf.Variable(True,
                                           dtype=tf.bool,
                                           trainable=False,
                                           name='training')
        random.seed()
        if hasattr(tf, 'random'):
            tf.random.set_random_seed(random.randint(-10000000, 10000000))
        else:
            tf.set_random_seed(random.randint(-10000000, 10000000))
        return

    def __del__(self):
        self.texecutor.shutdown()
        self.pexecutor.shutdown()
        return

    def get_training_tensor(self):
        ''' 学習中か否かを表すtraining tensorを返す '''
        return self.training_tensor

    def logger(self):
        from .logger import get_default_logger
        return get_default_logger()

kernel = Kernel()

name = 'kernel'

