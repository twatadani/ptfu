''' model.py - 学習モデルを記述するモジュール '''

class Model:
    ''' 各種学習モデルの規定となるクラス '''

    def __init__(self):
        self.gstep = None
        self.trainhooks = []
        self.validationstep = None
        return

    def train(self, **options):
        ''' このモデルを訓練する。与えられるパラメータと返り値は具象クラスによって異なる '''
        raise NotImplementedError

    def validate(self, **options):
        ''' 訓練したモデルに新しいデータを与えて検証する。与えられるパラメータと返り値は具象クラスによって異なる '''
        raise NotImplementedError

    def global_step_tensor(self):
        ''' global stepを表すtensorを返す '''
        import tensorflow.train as train
        if self.gstep is None:
            self.gstep = train.get_global_step()
        if self.gstep is None:
            self.gstep = train.create_global_step()
        return self.gstep

    def validation_step_tensor(self):
        ''' validation stepを表すtensorを返す '''
        import tensorflow as tf
        if self.validationstep is None:
            self.validationstep = tf.Variable(initial_value = 0,
                                              dtype = tf.int64,
                                              trainable = False,
                                              name = 'validation_step')
        return self.validationstep

    def register_trainhook(self, smartsessionhook):
        ''' 学習時のhookを登録する。ここで登録されたhookはtrainの際に自動的に呼び出される '''
        from ..smartsessionhook import SmartSessionHook
        if isinstance(smartsessionhook, SmartSessionHook):
            self.trainhooks.append(smartsessionhook)
