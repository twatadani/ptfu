''' TensorFlowのSessionを多機能化するモジュール '''

import tensorflow as tf
import os.path
from concurrent.futures import ThreadPoolExecutor

from .smartsessionhook import SmartSessionHook
from .loopsmartsessionhook import LoopSmartSessionHook
from .onetimesmartsessionhook import OneTimeSmartSessionHook

class SmartSession:
    ''' TensorflowのSessionと同様のinterfaceで多機能化したクラス '''

    def __init__(self, tfconfig, session_initialization_ops=None, initialization_feed_dict=None):
        ''' イニシャライザ TFConfigクラスのインスタンスを引数に与える '''
        from .loopsmartsessionhook import LoopSmartSessionHook

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
                LoopSmartSessionHook(hook_func = self._write_summary,
                                     hook_step = self.tfconfig.summary_save_interval,
                                     hook_mod = 1,
                                     synchronous = False,
                                     hook_name = 'write_summary'))

        if self.tfconfig.use_checkpoint:
            initial_hooks.append(
                LoopSmartSessionHook(hook_func = self._save_checkpoint,
                                     hook_step = self.tfconfig.checkpoint_save_interval,
                                     hook_mod = 0,
                                     synchronous = True,
                                     hook_name='save_checkpoint'))

        self.registerHooks(initial_hooks)

        self.last_fetches = None
        self.last_tensorvaluedict_endflag = None
        self.endflag_tensor_set = set()

        # セッション初期化に必要なop
        self.initialization_ops = []
        self.initialization_fd = None

        if session_initialization_ops is not None:
            self._register_initialize_ops(session_initialization_ops,
                                          initialization_feed_dict)

        self.initial_hooks_toberun = False
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
            self.session.run([tf.global_variables_initializer(),
                              tf.local_variables_initializer()])
            if len(self.initialization_ops) > 0:
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
            self.initial_hooks_toberun = True
            #self.run_initial_or_final_hooks(True)
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        ''' with構文終了時の処理 '''
        if self.session is not None:
            #self.run_initial_or_final_hooks(False) # modelに委譲
            self.session.close()

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

        # initial hookを走らせる → modelに委譲
        #if run_hooks == True and self.initial_hooks_toberun == True:
            #self.run_initial_or_final_hooks(True) 
            #self.initial_hooks_toberun = False

        # 呼び出し側から与えられた評価対象に自動で評価する対象を追加する
        finalfetches = set()
        is_fetch_list = False
        if isinstance(fetches, list) or isinstance(fetches, tuple): #リストorタプルの場合
            finalfetches |= set(fetches)
            is_fetch_list = True
        else: # それ以外の場合
            finalfetches.add(fetches)
            is_fetch_list = False

        return_fetches = list(finalfetches)

        # endflag用のfetchesを加える
        finalfetches |= self.endflag_tensor_set

        # その他SmartSession内で使用するtensorを加える
        finalfetches |= set(self.fetches_extended)

        # hooksで取得するtensorを加える
        for hook in self.hooks:
            finalfetches |= set(hook.tensorlist)

        # key=tensor, value=tensorの辞書を作る
        tensortensordict = { x: x for x in finalfetches }

        # 実際にSessionによる評価を行う
        self.last_tensorvaluedict = self.session.run(tensortensordict,
                                                     feed_dict,
                                                     options,
                                                     run_metadata)

        # 最新のglobal stepを保存
        self.last_global_step = self.last_tensorvaluedict[self.gstep_tensor]

        # 最新のsummaryを保存
        if self.tfconfig.use_summary:
            if self.merged is not None:
                self.last_summary = self.last_tensorvaluedict[self.merged]
            else:
                self.last_summary = None
            
        # endflag用
        self.last_tensorvaluedict_endflag = self._extract_tensorvaluedict(
            self.last_tensorvaluedict, self.endflag_tensor_set)

        # 次は返り値用
        self.last_tensorvaluedict_user = self._extract_tensorvaluedict(
            self.last_tensorvaluedict, return_fetches)

        if is_fetch_list:
            self.last_fetches = [ self.last_tensorvaluedict_user[x] for x in return_fetches ]
        else:
            self.last_fetches = self.last_tensorvaluedict_user[fetches]
        
        # stepに応じて登録されたhookを実行
        if run_hooks:
            self.run_loop_hooks(self.last_global_step, self.last_tensorvaluedict)

        return self.last_fetches

    def register_endflag_tensor(self, tensor):
        ''' このSmartSessionにEndFlagで使用するtensorを登録する '''
        self.endflag_tensor_set.add(tensor)

    def registerHooks(self, hook_list):
        '''step counter毎のhookを登録する
        hook_listは単体のhookオブジェクトまたはhookのリストまたはタプル形式でなくてはならない。
        個々のhookはSmartSessionHookのインスタンス '''

        if isinstance(hook_list, list) or isinstance(hook_list, tuple):
            if self.hooks is None:
                self.hooks = hook_list
                self.executor = ThreadPoolExecutor()
            else:
                self.hooks.extend(hook_list)
        else:
            if self.hooks is None:
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

    def run_initial_or_final_hooks(self, initial=True, feed_dict=None, options=None,
                                   run_metadata=None):
        ''' 初期hookを実行する 
        initial: Trueの場合initial hookを、Falseの場合final hookを実行する '''
        judge_func = lambda hook: (hook.run_at_startup and self.initial_hooks_toberun) if initial else hook.run_at_shutdown
        if (initial and self.initial_hooks_toberun) or initial == False:
            tensorset = set()
            for hook in self.hooks:
                if initial and hook.run_at_startup:
                    tensorset |= set(hook.tensorlist)
                elif (not initial) and hook.run_at_shutdown:
                    tensorset |= set(hook.tensorlist)
            tensortensordict = { x: x for x in tensorset }
            mastertensorvaluedict = self.session.run(tensortensordict,
                                                     feed_dict, options, run_metadata)
            self._run_hook_common(judge_func, mastertensorvaluedict)
            if initial == True:
                self.initial_hooks_toberun = False
        return

    def run_loop_hooks(self, global_step, mastertensorvaluedict):
        ''' loop hookを実行する '''
        judge_func = lambda hook: (isinstance(hook, LoopSmartSessionHook) and \
                                   hook.mod == global_step % hook.step) or \
            (isinstance(hook, OneTimeSmartSessionHook) and hook.step == global_step)
        self._run_hook_common(judge_func, mastertensorvaluedict)
        return

    def _run_hook_common(self, hook_judge_func, mastertensorvaluedict):
        ''' run_loop_hookとrun_initial_or_final_hookの共通部分を記載 '''
        from concurrent.futures import wait
        from .logger import get_default_logger
        logger = get_default_logger()

        futures = []
        for hook in self.hooks:
            if hook_judge_func(hook) == True:
                tensorvaluedict = self._extract_tensorvaluedict(mastertensorvaluedict,
                                                                hook.tensorlist)
                if hook.sync:
                    futures.append(self._run_hook_impl(hook, tensorvaluedict))
                else:
                    future = self._run_hook_impl(hook, tensorvaluedict)
                    self.executor.submit(self.__class__._run_future_exc_print, future, logger)
                if len(futures) > 0:
                    wait(futures)
                    for future in futures:
                        exc = future.exception()
                        if exc is not None:
                            logger.error(str(exc))
        return

    @staticmethod
    def _run_future_exc_print(future, logger):
        ''' 与えられたfutureが完了するのを待ち、例外が発生していたらプリントする '''
        from concurrent.futures import wait
        wait(future)
        exc = future.exception()
        if exc is not None:
            logger.error(str(exc))
        return

    @staticmethod
    def _create_tensortensordict(hooks, mode=SmartSessionHook.LOOP):
        ''' hooksから必要とされるtensor-tensor dictを作成する '''
        hooktype = None
        if mode == SmartSessionHook.ONETIME:
            hoooktype = OneTimeSmartSessionHook
        elif mode == SmartSessionHook.LOOP:
            hooktype = LoopSmartSessionHook
        else:
            hooktype = SmartSessionHook
        ttset = set()
        for hook in hooks:
            if isinstance(hook, hooktype):
                ttset = ttset | set(hook.tensorlist)
        return { x: x for x in ttset }

    @staticmethod
    def _extract_tensorvaluedict(masterdict, tensorlist):
        ''' { tensor: value, ...} 形式のmasterdictからtensorlistに含まれるものだけを抽出する '''
        return { x: masterdict[x] for x in tensorlist }

    def _run_hook_impl(self, hook, tensorvaluedict):
        return self.executor.submit(hook, tensorvaluedict)
        

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

name = 'smartsession'


            
        
        
