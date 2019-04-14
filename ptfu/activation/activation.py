''' activation.py - 自作活性化関数の基底クラスを記述する '''

class ActivationFunction:
    ''' 自作活性化関数の基底クラス '''

    def __init__(self, funcname):
        self.name = funcname

    def __call__(self, input_tensor):
        ''' 活性化関数のスケルトン input_tensorを受け取り、returnで結果のtensorを返す '''
        raise NotImplementedError

name = 'activation'
