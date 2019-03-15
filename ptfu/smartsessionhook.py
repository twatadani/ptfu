''' smartsessionhook.py: SmartSessionの実行中に併せて実行されるhookを記述する '''

class SmartSessionHook:
    ''' SmartSessionの学習ループ中に実行されるhook関数を表すクラス '''

    # class variable
    ALL = 0
    ONETIME = 1
    LOOP = 2

    def __init__(self, hook_func=None, synchronous=True,
                 required_tensor_list=None, 
                 feed_dict=None, hook_name=None, 
                 self_or_cls = None, **funcoptions):
        '''
        hook_func: 呼び出される関数。hook_funcはhook_func(tensorvaluedict)の形式で呼び出される。
        synchronous: Trueの場合同一スレッドで実行される。Falseの場合新しいスレッドで実行される。
        required_tensor_list: このhook実行に必要なtensorのリスト 
        feed_dict: このhook実行時のみ与える一時的なfeed_dict
        self_or_cls: hook関数がクラスメソッドあるいはインスタンスメソッドであった場合、selfまたはclsとして与えなければいけないパラメータを指定する
        funcoptions: hook_funcに与えるオプション
        '''
        self.func = hook_func if hook_func is not None else self.dummy

        self.sync = synchronous
        if required_tensor_list is None:
            self.tensorlist = []
        else:
            self.tensorlist = required_tensor_list
        if hook_name is not None:
            self.hookname = hook_name

        self.self_or_cls = self_or_cls

        if funcoptions is not None:
            self.funcoptions = funcoptions
            
        self.run_at_startup = False
        self.run_at_shutdown = False

        return

    def __call__(self, tensorvaluedict):
        ''' hook_funcを呼び出す '''
        if self.tensorlist is None or len(self.tensorlist) == 0:
            if self.funcoptions is None:
                if self.self_or_cls is None:
                    return self.func()
                else:
                    return self.func(self.self_or_cls)
            
            else:
                if self.self_or_cls is None:
                    return self.func(**self.funcoptions)
                else:
                    return self.func(self.self_or_cls, **self.funcoptions)
        else:
            if self.funcoptions is None:
                return self.func(tensorvaluedict)
            else:
                return self.func(tensorvaluedict, **self.funcoptions)

    def set_as_initial_hook(self):
        ''' スタートアップ時にこのhookを実行するよう設定する '''
        self.run_at_startup = True
        return


    def set_as_final_hook(self):
        ''' 終了時にこのhookを実行するよう設定する '''
        self.run_at_shutdown = True
        return
            
    def dummy(self):
        ''' 何もしないダミー関数 '''
        return


name = 'smartsessionhook'
