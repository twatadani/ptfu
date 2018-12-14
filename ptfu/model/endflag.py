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

    def reason(self):
        ''' should_endがTrueの場合、その理由をstrで返す。
        should_endがFalseの場合の動作は定めていない'''
        raise NotImplementedError

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

    def reason(self):
        ''' 学習終了の理由 '''
        s = 'Global stepが規定値に達したため。規定値 : '
        s += str(self.maxstep)
        s += ', global step: '
        s += str(self.smartsession.get_global_step())
        s += '.'
        return s

class LossNaNEndFlag(EndFlag):
    ''' loss関数がnanになったときに終了する条件 '''

    def __init__(self, loss_tensor, ncontinue=5):
        ''' loss_tensor: モニターするロス関数のtensor
        ncontinue: 何ステップ連続でnanになったら終了するか '''
        
        self.losslist = []
        self.losstensor = loss_tensor
        self.ncontinue = ncontinue

    def should_end(self):
        ''' 学習を終了すべきか返す '''
        import math
        # まず、 losslistを更新する
        newloss = self.smartsession.get_last_fetches[self.losstensor]
        self.losslist.append(newloss)
        if len(self.losslist) > self.ncontinue:
            self.losslist = self.losslist[1:]

        for loss in self.losslist:
            if not math.isnan(loss): # いずれか一つでもNaNでなければ、終了しない
                return False
        return True

    def reason(self):
        ''' 学習終了の理由 '''
        if not self.should_end():
            return 'LossNaNEndFlag: 終了条件に該当しません。'
        else:
            return str(ncontinue) + '回連続で損失関数がNaNとなったため'
        

class AndEndFlag(EndFlag):
    ''' 複数のEndFlagの&演算子による複合判定 '''

    def __init__(self, flag1, flag2):
        self.flag1 = flag1
        self.flag2 = flag2
        return

    def should_end(self):
        ''' 学習を終了すべきか返す '''
        return self.flag1.should_end() & self.flag2.should_end()

   def setSmartSession(self, smartsession):
       self.flag1.setSmartSession(smartsession)
       self.flag2.setSmartSession(smartsession)

    def reason(self):
        ''' 学習終了の理由 '''
        import os
        return self.flag1.reason() + ' かつ ' + os.linesep + self.flag2.reason()

class OrEndFlag(EndFlag):
    ''' 複数のEndFlagの|演算子による複合判定 '''

    def __init__(self, flag1, flag2):
        self.flag1 = flag1
        self.flag2 = flag2
        return

    def should_end(self):
        ''' 学習を終了すべきか返す '''
        return self.flag1.should_end() | self.flag2.should_end()

   def setSmartSession(self, smartsession):
       self.flag1.setSmartSession(smartsession)
       self.flag2.setSmartSession(smartsession)

    def reason(self):
        ''' 学習終了の理由 '''
        reason1 = self.flag1.reason()
        reason2 = self.flag2.reason()

        if self.flag1.should_end() and self.flag2.should_end():
            return reason1 + ' かつ ' + os.linesep + reason2
        elif self.flag1.should_end() and (not self.flag2.should_end()):
            return reason1
        elif (not self.flag1.should_end()) and self.flag2.should_end():
            return reason2
        else:
            return '終了条件を満たしていません'
            
            
