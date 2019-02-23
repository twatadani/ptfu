''' validationhook.py: Validationの際に処理を行うhookクラスを定義する '''

from ..smartsession import SmartSessionHook

class ValidationHook(SmartSessionHook):
    ''' Validationを行う際のHook関数を記述するためのクラス '''

    def __init__(self, hook_func=None, required_tensor_list=None, hook_name=None, **funcoptions):
        ''' ValidationHookの初期化。
        hook_func: 呼び出される関数。hook_func(tensorvaluedict, **funcoptions)の形式で呼び出される。
        required_tensor_list: このhookに必要なtensorのリスト
        funcoptions: hook_funcのオプション '''
        super(ValidationHook, self).__init__(hook_func=hook_func, hook_step=1, hook_mod=0, synchronous=True,
                                            required_tensor_list=required_tensor_list, hook_name=hook_name, **funcoptions)
        return

