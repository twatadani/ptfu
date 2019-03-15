''' TensorFlowのハードウェア関係などの動作を設定するモジュール '''

class TFConfig:
    ''' Tensorflowの動作設定を保存するクラス '''

    # class constant
    CPU = '/CPU:0'
    XLA_CPU = '/XLA_CPU:0'
    XLA_GPU = '/XLA_GPU:0'

    def __init__(self, **kwargs):
        ''' TFConfig イニシャライザ
        kwargsで指定されるパラメータは
        use_gpu: True or False # GPUを使用するかどうか デフォルトはTrue
        gpu_list: リスト(例: [0, 1, 2]) # 使用するGPUのリスト use_gpuがFalseの場合は無意味。デフォルトは検出されたGPUすべて
        gpu_parallelism: 自然数 # GPUの並列数。use_gpuがFalseの場合は無意味。デフォルトは1
        use_summary: True or False # TensorBoard用のSummaryを使用するかどうか デフォルトはFalse
        summary_save_interval: 自然数 # Summaryを保存する間隔。単位はglobal step。デフォルトは100
        use_checkpoint: True or False # SessionのCheckpointを保存するかどうか デフォルトはFalse
        checkpoint_save_interval: 自然数 # Checkpointを保存する間隔。単位はglobal step。デフォルトは1000
        summarydir: パス文字列 # summary, checkpointを保存するディレクトリ
        use_autoreload: True or False # 学習中断したときにcheckpointを読み込んで学習を再開するかどうか。デフォルトはTrue
        use_xla: XLA JITコンパイルを使用するかどうか
        '''

        # GPU関係の設定
        ## GPUを使用するかどうか
        self.use_gpu = True # デフォルト
        if 'use_gpu' in kwargs:
            self.use_gpu = kwargs['use_gpu']

        ## 使用するGPUのリスト
        self.gpu_list = None # デフォルトはNone
        if 'gpu_list' in kwargs:
            self.gpu_list = kwargs['gpu_list']

        ## GPU並列動作数
        self.gpu_parallelism = 1 # デフォルト
        if 'gpu_parallelism' in kwargs:
            self.gpu_parallelism = kwargs['gpu_parallelism']


        # Summary, Checkpoint保存関係の設定
        ## Summaryを使用するかどうか
        self.use_summary = False # デフォルト
        if 'use_summary' in kwargs:
            self.use_summary = kwargs['use_summary']

        ## Summaryを保存する間隔
        self.summary_save_interval = 100 # デフォルト
        if 'summary_save_interval' in kwargs:
            self.summary_save_interval = kwargs['summary_save_interval']

        ## Checkpointを使用するかどうか
        self.use_checkpoint = False # デフォルト
        if 'use_checkpoint' in kwargs:
            self.use_checkpoint = kwargs['use_checkpoint']

        ## Checkpointを保存する間隔
        self.checkpoint_save_interval = 1000 # デフォルト
        if 'checkpoint_save_interval' in kwargs:
            self.checkpoint_save_interval = kwargs['checkpoint_save_interval']

        ## summary, checkpointを保存するディレクトリの設定
        self.summarydir = '.' # デフォルトはカレントディレクトリ
        if 'summarydir' in kwargs:
            self.summarydir = kwargs['summarydir']

        ## 前回の学習を再開するかどうか
        self.use_autoreload = True # デフォルト
        if 'use_autoreload' in kwargs:
            self.use_autoreload = kwargs['use_autoreload']

        self.use_xla = False # デフォルト
        if 'use_xla' in kwargs:
            self.use_xla = kwargs['use_xla']

        # 並列実行のtowerを構築する
        self.towers = self._create_towers()

        return

    def create_configproto(self):
        ''' 設定情報からTensorFlowのConfigProtoを作成する '''
        import tensorflow as tf
        if self.use_gpu:
            gpu_options = tf.GPUOptions(
                visible_device_list = self._list2strlist(self.gpu_list),
                allow_growth = True) # メモリは最初から全確保でなく、必要に応じて確保する
            cp = tf.ConfigProto(
                gpu_options = gpu_options,
                allow_soft_placement = True)
            # XLA JITコンパイルを有効にする(GPUのみ)
            #if self.use_xla:
                #cp.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        else:
            # 環境変数でGPUを制限する
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            cp = tf.ConfigProto(
                allow_soft_placement = True)

        
        return cp


    @staticmethod
    def _list2strlist(gpu_list):
        '''gpu_list = [0, 1, 2, 3]形式のリストを
        '0, 1, 2, 3'の文字列に変換する'''
        strlist = ''
        for i, gpu in enumerate(gpu_list):
            strlist += str(gpu)
            if i != len(gpu_list)-1:
                strlist += ', '
        return strlist

    def ntowers(self):
        ''' 並列実行のタワー数を返す '''
        if not self.use_gpu:
            return 1
        else:
            return self.gpu_parallelism

    def _create_towers(self):
        ''' 設定された並列実行のtowerリストを作成する。
        tower = []のリストで、さらにリストの要素の1タワーは利用可能なデバイスのリスト '''

        towers = []
        for i in range(self.ntowers()):
            if not self.use_gpu: # CPU学習の場合
                if self.use_xla: # XLA_CPU
                    tower = [TFConfig.XLA_CPU]
                else:
                    tower = [TFConfig.CPU]
            else: # GPU学習の場合
                gpuprefix = '/GPU:'
                xlaprefix = '/XLA_GPU:'
                
                if self.use_xla:
                    prefix = xlaprefix
                else:
                    prefix = gpuprefix
            
                ngpu_per_tower = len(self.gpu_list)
                
                tw_start = i * ngpu_per_tower
                #print('tw_start = ', tw_start)
                tw_end = tw_start + ngpu_per_tower
                #print('tw_end = ', tw_end)
                if i == self.ntowers() - 1:
                    tw_end = len(self.gpu_list)

                tower = []
                for gpunum in self.gpu_list[tw_start:tw_end]:
                    tower.append(prefix + str(gpunum))
            towers.append(tower)
        return towers
        
