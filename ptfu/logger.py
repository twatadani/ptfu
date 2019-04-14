''' logger.py - ptfu用のロギングユーティリティー '''

import logging

class Logger:
    ''' ロギングを行うオブジェクト '''

    # クラス変数
    ptfu_default_logger = None

    def __init__(self, logdir=None):
        ''' loggerの初期化 '''
        import sys

        self.streamloggers = []
        defaultlogger = logging.getLogger('ptfu_default')

        if not defaultlogger.hasHandlers():
            shandler = logging.StreamHandler(sys.stdout)
            defaultlogger.addHandler(shandler)

        self.streamloggers.append(defaultlogger)
        defaultlogger.setLevel(1)

        self.fileloggers = []
        self.logdir = logdir
        if self.logdir is not None:
            from datetime import datetime
            import os.path
            import os
            filelogger = logging.getLogger('ptfu_filelogger')
            now = datetime.now()
            logfilename = 'log-'
            logfilename += str(now.year) + str(now.month)
            logfilename += str(now.day) + '-' + str(now.hour)
            logfilename += str(now.minute) + str(now.second) + '.txt'
            logfile = os.path.join(self.logdir, logfilename)
            if os.path.exists(logfile):
                os.remove(logfile)
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            fhandler = logging.FileHandler(logfile)
            filelogger.removeHandler(fhandler) # 二重登録を防ぐため
            filelogger.addHandler(fhandler)
            filelogger.setLevel(1)
            self.fileloggers.append(filelogger)
        return

    def setLevel(self, level):
        ''' logging setLevelと同様 '''
        for logger in self.streamloggers:
            logger.setLevel(level)
        for logger in self.fileloggers:
            logger.setLevel(level)
        return

    def _log(self, attr, msg, *args, **kwargs):
        ''' 内部用関数 '''

        termonly = False
        if 'terminalonly' in kwargs and kwargs['terminalonly'] == True:
            termonly = True
        try:
            for logger in self.streamloggers:
                func = getattr(logger, attr)
                func(msg, *args, **kwargs)
            if not termonly:
                for logger in self.fileloggers:
                    func = getattr(logger, attr)
                    func(msg, *args, **kwargs)
        except:
            import traceback
            traceback.print_exc()
        return

    def log(self, msg, *args, **kwargs):
        ''' 通常のロギング python loggerモジュールと同じインターフェースとして使う '''
        self._log('info', msg, *args, **kwargs)
        return

    def debug(self, msg, *args, **kwargs):
        self._log('debug', msg, *args, **kwargs)
        return

    def warning(self, msg, *args, **kwargs):
        self._log('warning', msg, *args, **kwargs)
        return

    def info(self, msg, *args, **kwargs):
        self._log('info', msg, *args, **kwargs)
        return

    def error(self, msg, *args, **kwargs):
        self._log('error', msg, *args, **kwargs)
        return

    def critical(self, msg, *args, **kwargs):
        self._log('critical', msg, *args, **kwargs)
        return

    def exception(self, msg, *args, **kwargs):
        self._log('exception', msg, *args, **kwargs)
        return

def get_default_logger():
    if Logger.ptfu_default_logger is None:
        Logger.ptfu_default_logger = Logger()
    return Logger.ptfu_default_logger

def set_default_logger(logger):
    Logger.ptfu_default_logger = logger
    return


name = 'logger'


