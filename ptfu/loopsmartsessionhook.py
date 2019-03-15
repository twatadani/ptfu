''' loopsmartsessionhook.py: ループ中に繰り返し実行されるSmartSessionHookを記述 '''

from .smartsessionhook import SmartSessionHook

class LoopSmartSessionHook(SmartSessionHook):
    ''' 繰り返し実行されるSmartSessionHook '''

    def __init__(self, hook_func=None, hook_step=1, hook_mod=0, synchronous=True,
                 required_tensor_list=None, feed_dict=None, hook_name=None, 
                 self_or_cls=None, **funcoptions):
        '''
        hook_func: 呼び出される関数。hook_funcはhook_func(tensorvaluedict)の形式で呼び出される。
        hook_step: 何ステップ毎にhook_funcが呼び出されるか
        hook_mod: step % hook_step = hook_modの時に呼び出される
        synchronous: Trueの場合同一スレッドで実行される。Falseの場合新しいスレッドで実行される。
        required_tensor_list: このhook実行に必要なtensorのリスト
        feed_dict: このhook実行時のみ与える一時的なfeed_dict
        self_or_cls: hook関数がクラスメソッドあるいはインスタンスメソッドであった場合、selfまたはclsとして与えなければいけないパラメータを指定する
        funcoptions: hook_funcに与えるオプション
        '''
        super(LoopSmartSessionHook, self).__init__(
            hook_func, synchronous, required_tensor_list, feed_dict, hook_name, 
            self_or_cls, **funcoptions)
        self.step = hook_step
        self.mod = hook_mod
        return


name = 'loopsmartsessionhook'
