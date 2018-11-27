# -*- coding: utf-8 -*-

''' TensorFlowのSessionを多機能化するモジュール '''

import tensorflow as tf
import os.path
from concurrent.futures import ThreadPoolExecutor

class SmartSession:
    ''' TensorflowのSessionと同様のinterfaceで多機能化したクラス '''

    def __init__(self, tfconfig):
        ''' イニシャライザ TFConfigクラスのインスタンスを引数に与える '''

        self.tfconfig = tfconfig
        self.last_global_step = 0
        self.session = None
        self.saver = None
        self.gstep_tensor = tf.train.get_global_step()
        self.executor = None
        self.fetches_extended = [self.gstep] # sessionで評価する対象に加えるリスト
        self.hooks = None
        initial_hooks = []

        if self.tfconfig.use_summary:
            self.merged = tf.summary.merge_all()
            self.fetches_extended.append(self.merged)
            self.last_summary = None
            initial_hooks.append(
                [self.tfconfig.summary_save_interval, 1, False, self._write_summary])

        if self.tfconfig.use_checkpoint:
            initial_hooks.append(
                [self.tfconfig.checkpoint_save_interval, 0, True, self._save_checkpoint])

        if len(initial_hooks) > 0:
            self.registerHooks(initial_hooks)
        return
        

    def __enter__(self):
        '''with構文を使ってSessionを開く'''

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
            lastchkp = tf.train.latest_checkpoint(self.tfh.chkpdir)

            # restore前に一旦初期化する
            self.session.run(
                [tf.global_variables_initializer(),
                 tf.local_variables_initializer()])

            if lastchkp is not None:
                self.saver.restore(self.session,
                                   save_path=lastchkp)
                chkp_loaded = True
                self.last_global_step = self.session.run(self.gstep)
        
        if not chkp_loaded: # 再開しない設定、または読み込めなかったとき
            self.run_hooks(0)
        return self


    def __exit__(self, exc_type, exc_value, tracebabck):
        ''' with構文終了時の処理 '''
        if self.session is not None:
            self.session.close()

        if exc_type is None: #例外なしで終了したとき
            return
        else:
            return False


    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        '''Session.runへのwrapper'''

        # 呼び出し側から与えられた評価対象に自動で評価する対象を追加する
        is_fetches_iterable = hasattr(fetches, '__iter__')
        finalfetches = None
        if is_fetches_iterable:
            fetches.extend(self.fetches_extended)
            finalfetches = fetches
        else:
            finalfetches = [fetches]
            finalfetches.extend(self.fetches_extended)

        # 実際にSessionによる評価を行う
        result = self.session.run(finalfetches,
                                  feed_dict,
                                  options,
                                  run_metadata)

        # 最新のglobal stepを保存
        gstep_idx = len(result) - len(self.fetches_extended)
        self.last_global_step = result[gstep_idx]

        # 最新のsummaryを保存
        if self.tfconfig.use_summary:
            sum_idx = gstep_idx + 1
            self.last_summary = result[sum_idx]

        # stepに応じて登録されたhookを実行
        self.run_hooks(self.last_global_step)

        # 自動で追加した対象を削除して呼び出し側に結果を返す
        if is_fetches_iterable:
            return result[0:gstep_idx]
        else:
            return result[0]

    def registerHooks(self, hook_list):
        '''step counter毎のhookを登録する
        hook_listはhookのリストまたはタプル形式でなくてはならない。
        個々のhookは
        (何ステップ数ごとに呼び出されるか,
        ステップ数のmodulo, 
        SynchronousかAsynchronousか(Sync: True, Async: False),
        呼び出される関数、その引数リスト)の形式
        Synchronousなものは同一スレッド内で、Asynchronousなものは
        新しいスレッドで実行される'''

        if self.hooks is None:
            self.hooks = hook_list
            self.executor = ThreadPoolExecutor()
        else:
            self.hooks.extend(hook_list)
        return

    def get_global_step(self):
        ''' 最新のglobal stepの値を返す '''
        return self.last_global_step


    def _write_summary(self):
        ''' summary書き出し用のhook関数 '''
        self.summarywriter.add_summary(self.last_summary, self.last_global_step)
        self.summarywriter.flush()
        return

    def _save_checkpoint(self):
        ''' gstep時点のcheckpointをセーブするhook関数 '''
        savedir = self.tfconfig.summarydir
        prefix = os.path.join(savedir, 'model.ckpt')
        result = self.saver.save(self.session, prefix, self.last_global_step)
        return result
        
        
