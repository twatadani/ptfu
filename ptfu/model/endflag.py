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

class TensorValueEndFlag(EndFlag):
    ''' 特定の1つのTensorの値を使って終了判定を行う終了条件
    このクラスはabstract class '''
    
    def __init__(self, tensor, ncontinue = 5):
        ''' ncontinue: 何回分の値を使うか '''
        super(TensorValueEndFlag, self).__init__()
        self.tensor = tensor
        self.ncontinue = ncontinue
        self.lastvalues = []

    def setSmartSession(self, smartsession):
        ''' SmartSessionを設定する '''
        self.smartsession = smartsession
        self.smartsession.register_endflag_tensor(self.tensor)
        return

    def update_lastvalues(self):
        if self.smartsession.last_tensorvaluedict_endflag is not None:
            lastvalue = self.smartsession.last_tensorvaluedict_endflag[self.tensor]
            self.lastvalues.append(lastvalue)
        
            # リストの長さを調節する
            if len(self.lastvalues) > self.ncontinue:
                self.lastvalues = self.lastvalues[(len(self.lastvalues)-self.ncontinue):]
        return

class LossNaNEndFlag(TensorValueEndFlag):
    ''' loss関数がnanになったときに終了する条件 '''

    def __init__(self, loss_tensor, ncontinue=5):
        ''' loss_tensor: モニターするロス関数のtensor
        ncontinue: 何ステップ連続でnanになったら終了するか '''
        super(LossNaNEndFlag, self).__init__(loss_tensor, ncontinue)

    def should_end(self):
        ''' 学習を終了すべきか返す '''
        import math
        # まず、 lossのリストを更新する
        self.update_lastvalues()

        if len(self.lastvalues) == 0:
            return False

        for loss in self.lastvalues:
            if not math.isnan(loss): # いずれか一つでもNaNでなければ、終了しない
                return False
        return True

    def reason(self):
        ''' 学習終了の理由 '''
        if not self.should_end():
            return 'LossNaNEndFlag: 終了条件に該当しません。'
        else:
            return str(self.ncontinue) + '回連続で損失関数がNaNとなったため'

class TensorSmallerEndFlag(TensorValueEndFlag):
    ''' あるtensorの値が一定より下回ったら終了する条件 '''

    def __init__(self, tensor, threshold, ncontinue=5):
        ''' tensor: 監視するtensor
        threshold: この値を下回ったら終了
        ncontinue: 最新n回の平均値をthresholdと比較する '''
        
        super(TensorSmallerEndFlag, self).__init__(tensor, ncontinue)
        self.threshold = threshold

    def should_end(self):
        self.update_lastvalues()

        if len(self.lastvalues) == 0:
            return False

        return self.calculate_tensorsum() <= self.threshold * self.ncontinue
        
    def calculate_tensorsum(self):
        tensorsum = 0
        for tensorvalue in self.lastvalues:
            tensorsum += tensorvalue
        return tensorsum

    def reason(self):
        if self.should_end():
            return str(self.tensor) + 'の値が' + str(self.threshold) + '以下となったため。value=' + \
                str(self.calculate_tensorsum() / self.ncontinue)
        else:
            return '終了条件に該当しません。'

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
        return

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
        return

    def reason(self):
        ''' 学習終了の理由 '''
        import os
        reason1 = self.flag1.reason()
        reason2 = self.flag2.reason()

        if self.flag1.should_end() and self.flag2.should_end():
            return reason1 + ' かつ ' + os.linesep + reason2
        elif self.flag1.should_end() and (not self.flag2.should_end()):
            return reason1
        elif (not self.flag1.should_end()) and self.flag2.should_end():
            return reason2
        else:
            return '終了条件を満たしていません。 (may be bug?)'

name = 'endflag'

            
            
