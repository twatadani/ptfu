''' onetimesmartsessionhook.py: 一度だけ実行されるSmartSessionHookを記述 '''

from .smartsessionhook import SmartSessionHook

class OneTimeSmartSessionHook(SmartSessionHook):
    ''' 繰り返し実行されるSmartSessionHook '''

    def __init__(self, hook_func=None, hook_step=1, synchronous=True,
                 required_tensor_list=None, feed_dict=None, hook_name=None,
                 self_or_cls=None, **funcoptions):
        '''
        hook_func: 呼び出される関数。hook_funcはhook_func(tensorvaluedict)の形式で呼び出される。
        hook_step: global_stepがいくつのときにこのhookが実行されるか。initial, finalのみの場合は-1などを指定
        synchronous: Trueの場合同一スレッドで実行される。Falseの場合新しいスレッドで実行される。
        required_tensor_list: このhook実行に必要なtensorのリスト
        feed_dict: このhook実行時のみ与える一時的なfeed_dict
        self_or_cls: hook関数がクラスメソッドあるいはインスタンスメソッドであった場合、selfまたはclsとして与えなければいけないパラメータを指定する
        funcoptions: hook_funcに与えるオプション
        '''
        super(OneTimeSmartSessionHook, self).__init__(
            hook_func, syncronous, required_tensor_list, feed_dict, hook_name, 
            self_or_cls, **funcoptions)
        self.step = hook_step
        return

name = 'loopsmartsessionhook'
