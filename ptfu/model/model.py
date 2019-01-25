''' model.py - 学習モデルを記述するモジュール '''

class Model:
    ''' 各種学習モデルの規定となるクラス '''

    def __init__(self):
        self.gstep = None
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
