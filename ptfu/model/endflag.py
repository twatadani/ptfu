''' endflag.py - 学習の終了条件を記述するモジュール '''

class EndFlag:
    ''' 学習の終了条件を表す基底クラス '''

    def __init__(self):
        self.smartsession = None
        return

    def should_end(self):
        ''' 学習を終了すべきかどうかを返す '''
        raise NotImplementedError

    def setSmartSession(self, smartsession):
        self.smartsession = smartsession
        return

    def __and__(self, other):
        ''' &演算子のオーバーロード '''
        return AndEndFlag(self, other)

    def __or__(self, other):
        ''' |演算子のオーバーロード '''
        return OrEndFlag(self, other)
        
class NoneEndFlag(EndFlag):
    ''' ダミー用、永遠に終了しないEndFlag '''

    def should_end(self, **options):
        return False


class MaxGlobalStepFlag(EndFlag):
    ''' global step数が一定に達すると終了する条件 '''

    def __init__(self, max_global_step):
        super(MaxGlobalStepFlag, self).__init__()
        assert max_global_step > 0
        self.maxstep = max_global_step

    def should_end(self):
        ''' 学習を終了すべきかを返す '''
        if self.smartsession is None:
            return False
        else:
            gstep = self.smartsession.get_global_step()
            return gstep >= self.maxstep

class AndEndFlag(EndFlag):
    ''' 複数のEndFlagの&演算子による複合判定 '''

    def __init__(self, flag1, flag2):
        self.flag1 = flag1
        self.flag2 = flag2
        return

    def should_end(self):
        ''' 学習を終了すべきか返す '''
        return self.flag1.should_end() & self.flag2.should_end()

class OrEndFlag(EndFlag):
    ''' 複数のEndFlagの|演算子による複合判定 '''

    def __init__(self, flag1, flag2):
        self.flag1 = flag1
        self.flag2 = flag2
        return

    def should_end(self):
        ''' 学習を終了すべきか返す '''
        return self.flag1.should_end() | self.flag2.should_end()
