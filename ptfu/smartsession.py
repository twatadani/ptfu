''' TensorFlowのSessionを多機能化するモジュール '''

import tensorflow as tf
import os.path
from concurrent.futures import ThreadPoolExecutor

class SmartSession:
    ''' TensorflowのSessionと同様のinterfaceで多機能化したクラス '''

    def __init__(self, tfconfig, session_initialization_ops=None, initialization_feed_dict=None):
        ''' イニシャライザ TFConfigクラスのインスタンスを引数に与える '''

        self.tfconfig = tfconfig
        self.last_global_step = 0
        self.session = None
        self.saver = None
        self.gstep_tensor = tf.train.get_global_step()
        if self.gstep_tensor is None:
            self.gstep_tensor = tf.train.create_global_step()
        self.executor = None
        self.fetches_extended = [self.gstep_tensor] # sessionで評価する対象に加えるリスト
        self.hooks = None
        initial_hooks = []

        if self.tfconfig.use_summary:
            self.merged = tf.summary.merge_all()
            if self.merged is not None:
                self.fetches_extended.append(self.merged)
            self.last_summary = None
            initial_hooks.append(
                SmartSessionHook(hook_func = self._write_summary,
                                 hook_step = self.tfconfig.summary_save_interval,
                                 hook_mod = 1,
                                 synchronous = False,
                                 hook_name = 'write_summary'))

        if self.tfconfig.use_checkpoint:
            initial_hooks.append(
                SmartSessionHook(hook_func = self._save_checkpoint,
                                 hook_step = self.tfconfig.checkpoint_save_interval,
                                 hook_mod = 0,
                                 synchronous = True,
                                 hook_name='save_checkpoint'))

        #if len(initial_hooks) > 0:
        self.registerHooks(initial_hooks)

        self.last_fetches = None
        self.last_tensorvaluedict_endflag = None
        self.endflag_tensor_set = set()

        # セッション初期化に必要なop
        self.initialization_ops = [
            tf.global_variables_initializer(),
            tf.local_variables_initializer() ]
        self.initialization_fd = None

        if session_initialization_ops is not None:
            self._register_initialize_ops(session_initialization_ops,
                                          initialization_feed_dict)

        return
        

    def __enter__(self):
        '''with構文を使ってSessionを開く'''
        from . import logger

        # loggerを取得
        logger = logger.get_default_logger()

        # Sessionを作成
        self.session = tf.Session(config = self.tfconfig.create_configproto())

        # Summaryを保存する設定の場合、summarywriterを準備する
        if self.tfconfig.use_summary:
            self.summarywriter = tf.summary.FileWriter(
                self.tfconfig.summarydir, graph = self.session.graph)
        
        # 学習再開をする設定の場合、最新のCheckpointを読み込む
        chkp_loaded = False
        if self.tfconfig.use_autoreload:
            if self.saver is None:
                self.saver = tf.train.Saver()
            lastchkp = tf.train.latest_checkpoint(self.tfconfig.summarydir)

            # restore前に一旦初期化する
            self.session.run(self.initialization_ops,
                             feed_dict=self.initialization_fd)

            try:
                if lastchkp is not None:
                    self.saver.restore(self.session,
                                       save_path=lastchkp)
                    chkp_loaded = True
                    self.last_global_step = self.session.run(self.gstep_tensor)
                    logger.info('チェックポイントの自動復元に成功しました。')
            except:
                logger.info('チェックポイントの自動復元に失敗しました。新しく学習を開始します。')
                chkp_loaded = False
        
        if not chkp_loaded: # 再開しない設定、または読み込めなかったとき
            self.run_hooks(0)
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        ''' with構文終了時の処理 '''
        if self.session is not None:
            self.session.close()
            #self.session.__exit__(exc_type, exc_value, traceback)

        if exc_type is None: #例外なしで終了したとき
            return
        else:
            return False

    def _register_initialize_ops(self, initialization_ops, feed_dict=None):
        ''' セッション初期化に必要なopを登録する '''
        if isinstance(initialization_ops, list) or isinstance(initialization_ops, tuple):
            self.initialization_ops.extend(initialization_ops)
        else:
            self.initialization_ops.append(initialization_ops)
        self.initialization_fd = feed_dict
        return

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None, run_hooks=True):
        '''Session.runへのwrapper
        run_hooks: このrunでhookを実行するかどうか。hookを実行したくない場合はFalseにする '''

        # 呼び出し側から与えられた評価対象に自動で評価する対象を追加する
        finalfetches = set()
        is_fetch_list = False
        if isinstance(fetches, list) or isinstance(fetches, tuple): #リストorタプルの場合
            finalfetches |= set(fetches)
            is_fetch_list = True
            #finalfetches.extend(fetches)
        else: # それ以外の場合
            finalfetches.add(fetches)
            is_fetch_list = False

        # endflag用のfetchesを加える
        finalfetches |= self.endflag_tensor_set

        # その他SmartSession内で使用するtensorを加える
        finalfetches |= set(self.fetches_extended)
        
        # key-tensor, value=indexの逆引き辞書を作る
        actual_fetches = list(finalfetches)
        reversedict = {}
        for i, tensor in enumerate(actual_fetches):
            reversedict[tensor] = i

        # 実際にSessionによる評価を行う
        result = self.session.run(actual_fetches,
                                  feed_dict,
                                  options,
                                  run_metadata)



        # 結果をdictに整理して保存する。
        # 全部
        self.last_tensorvaluedict = {}
        for tensor in actual_fetches:
            self.last_tensorvaluedict[tensor] = result[reversedict[tensor]]

        # 最新のglobal stepを保存
        self.last_global_step = self.last_tensorvaluedict[self.gstep_tensor]
        #gstep_idx = len(result) - len(self.fetches_extended)
        #self.last_global_step = result[gstep_idx]

        # 最新のsummaryを保存
        if self.tfconfig.use_summary:
            if self.merged is not None:
                self.last_summary = self.last_tensorvaluedict[self.merged]
            else:
                self.last_summary = None
            
            #sum_idx = gstep_idx + 1
            # summaryに保存するものがなにもない場合の対策
            #if len(result) >= sum_idx+1:
                #self.last_summary = result[sum_idx]
            #else:
                #self.last_summary = None



        # endflag用
        self.last_tensorvaluedict_endflag = {}
        for tensor in self.endflag_tensor_set:
            self.last_tensorvaluedict_endflag[tensor] = result[reversedict[tensor]]

        # 次は返り値用
        #self.last_tensorvaluedict_user = {}
        if is_fetch_list:
            self.last_fetches = []
            for tensor in fetches:
                self.last_fetches.append(result[reversedict[tensor]])
                #self.last_tensorvaluedict_user[tensor] = result[reversedict[tensor]]
        else:
            self.last_fetches = result[reversedict[fetches]]
            #self.last_tensorvaluedict_user[fetch] = result[reversedict[fetch]]
        
        # stepに応じて登録されたhookを実行
        if run_hooks:
            self.run_hooks(self.last_global_step, feed_dict=feed_dict,
                           options=options, run_metadata=run_metadata)

        return self.last_fetches

    def register_endflag_tensor(self, tensor):
        ''' このSmartSessionにEndFlagで使用するtensorを登録する '''
        self.endflag_tensor_set.add(tensor)

    def registerHooks(self, hook_list):
        '''step counter毎のhookを登録する
        hook_listは単体のhookオブジェクトまたはhookのリストまたはタプル形式でなくてはならない。
        個々のhookはSmartSessionHookのインスタンス '''

        if isinstance(hook_list, list) or isinstance(hook_list, tuple):
            print('list mode')
            if self.hooks is None:
                print('self.hooks is None')
                self.hooks = hook_list
                self.executor = ThreadPoolExecutor()
            else:
                self.hooks.extend(hook_list)
        else:
            print('single mode')
            if self.hooks is None:
                print('self.hooks is None')
                self.hooks = [hook_list]
                self.executor = ThreadPoolExecutor()
            else:
                self.hooks.append(hook_list)
        return

    def get_global_step(self):
        ''' 最新のglobal stepの値を返す '''
        return self.last_global_step
    
    def get_last_fetches_dict(self):
        ''' 最新のrunの結果を返す '''
        return self.last_fetches_dict

    def run_hooks(self, global_step, feed_dict=None, options=None, run_metadata=None):

        for hook in self.hooks:

            tensorvaluedic = {}
            if len(hook.tensorlist) > 0:
                values = self.session.run(hook.tensorlist, feed_dict, options, run_metadata)
            else:
                values = []
            for i, tensor in enumerate(hook.tensorlist):
                tensorvaluedic[tensor] = values[i]

            if hook.mod == global_step % hook.step:
                if hook.sync:
                    hook(tensorvaluedic)
                else:
                    future = self.executor.submit(hook, tensorvaluedic)
        return

    def _write_summary(self):
        ''' summary書き出し用のhook関数 '''
        if self.last_summary is not None:
            try:
                self.summarywriter.add_summary(self.last_summary, self.last_global_step)
                self.summarywriter.flush()
            except:
                import traceback
                from . import get_default_logger
                logger = get_default_logger()
                logger.warning('Summaryの書き出し中にエラーが発生しました。')
                traceback.print_exc()
                logger.warning('無視して続行します。')
                
        return

    def _save_checkpoint(self):
        ''' gstep時点のcheckpointをセーブするhook関数 '''
        savedir = self.tfconfig.summarydir
        prefix = os.path.join(savedir, 'model.ckpt')
        try:
            result = self.saver.save(self.session, prefix, self.last_global_step)
            return result
        except:
            import traceback
            from . import get_default_logger
            logger = get_default_logger()
            logger.warning('Checkpointのセーブ中にエラーが発生しました。')
            traceback.print_exc()
            logger.warning('続行します。')
            return

class SmartSessionHook:
    ''' SmartSessionの学習ループ中に実行されるhook関数を表すクラス '''

    def __init__(self, hook_func=None, hook_step=1, hook_mod=0, synchronous=True,
                 required_tensor_list=None, hook_name=None, **funcoptions):
        '''
        hook_func: 呼び出される関数。hook_funcはhook_func(tensorvaluedict)の形式で呼び出される。
        hook_step: 何ステップ毎にhook_funcが呼び出されるか
        hook_mod: step % hook_step = hook_modの時に呼び出される
        synchronous: Trueの場合同一スレッドで実行される。Falseの場合新しいスレッドで実行される。
        required_tensor_list: このhook実行に必要なtensorのリスト 
        funcoptions: hook_funcに与えるオプション
        '''
        self.func = hook_func if hook_func is not None else self.dummy
        self.step = hook_step
        self.mod = hook_mod
        self.sync = synchronous
        if required_tensor_list is None:
            self.tensorlist = []
        else:
            self.tensorlist = required_tensor_list
        if hook_name is not None:
            self.hookname = hook_name

        if funcoptions is not None:
            self.funcoptions = funcoptions
        return

    def __call__(self, tensorvaluedict):
        ''' hook_funcを呼び出す '''
        if self.tensorlist is None or len(self.tensorlist) == 0:
            if self.funcoptions is None:
                return self.func()
            else:
                return self.func(**self.funcoptions)
        else:
            if self.funcoptions is None:
                return self.func(tensorvaluedict)
            else:
                return self.func(tensorvaluedict, **self.funcoptions)

            
    def dummy(self):
        ''' 何もしないダミー関数 '''
        return
            
        
        
