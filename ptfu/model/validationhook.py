''' validationhook.py: Validationの際に処理を行うhookクラスを定義する '''

from ..loopsmartsessionhook import LoopSmartSessionHook

class ValidationHook(LoopSmartSessionHook):
    ''' Validationを行う際のHook関数を記述するためのクラス '''

    def __init__(self, hook_func=None, required_tensor_list=None, hook_name=None, 
                 self_or_cls=None, **funcoptions):
        ''' ValidationHookの初期化。
        hook_func: 呼び出される関数。hook_func(tensorvaluedict, **funcoptions)の形式で呼び出される。
        required_tensor_list: このhookに必要なtensorのリスト
        self_or_cls: hook関数がクラスメソッドあるいはインスタンスメソッドであった場合、selfまたはclsとして与えなければいけないパラメータを指定する
        funcoptions: hook_funcのオプション '''
        super(ValidationHook, self).__init__(hook_func=hook_func, hook_step=1, hook_mod=0, synchronous=True,
                                            required_tensor_list=required_tensor_list, hook_name=hook_name, self_or_cls=self_or_cls, **funcoptions)
        return

