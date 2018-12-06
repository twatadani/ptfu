# -*- coding: utf-8 -*-

''' DataSetCreatorモジュール: データセットの作成を行う '''

class DatasetCreator:

    @staticmethod
    def datasetCreatorFromCustomRW(srcreader, dstwriter):
        ''' srcreader, dstwriterを指定してDatasetCreatorのインスタンスを生成する '''
        instance = DatasetCreator(None, None, None, None, None, None)
        instance.srcreader = srcreader
        instance.dstwriter = dstwriter
        return instance

    def __init__(self,
                 srcdatatype,
                 srcstoretype,
                 srcpath,
                 dststoretype,
                 dstpath,
                 datasetname):
        ''' DatasetCreatorの初期化
        srcdatatype: 元データの画像形式。DataType enumから選択する
        srcstoretype: 元データのアーカイブ形式。StoreType enumから選択する
        srcpath: 元データの場所。ディレクトリまたはアーカイブファイルを指定する
        dststoretype: 作成するデータセットのアーカイブ形式。StoreType enumから選択する
        dstpath: 作成するデータセットの置き場所。ディレクトリで指定する。
        datasetname: 作成するデータセットの名前 '''
        from .srcreader import SrcReader
        from .dstwriter import DstWriter

        self.srcreader = None
        self.dstwriter = None

        if srcdatatype is not None and srcstoretype is not None and srcpath is not None:
            self.srcreader = SrcReader(srcdatatype, srcstoretype, srcpath)

        if dststoretype is not None and dstpath is not None and datasetname is not None:
            self.dstwriter = DstWriter(dststoretype, dstpath, datasetname)

        self.logfunc = None
        return

    def setSplitByGroups(self, n):
        ''' 書き込みのグループ数を指定して分割書き込みを行う '''
        self.dstwriter.setSplitByGroups(n)
        return

    def setSplitByGroupNumber(self, n):
        ''' 書き込みの分割を1グループあたりのデータ数で指定する '''
        self.dstwriter.setSplitByGroupNumber(n)
        return

    def setlogfunc(self, func):
        ''' ロギング用の関数を設定する '''
        self.logfunc = func
        
    def log(self, *args):
        ''' ログを出力する 特別なlogfuncが設定されていない場合は組み込みのprintを使用する'''
        logfunc = None
        if self.logfunc is None:
            logfunc = print
        else:
            logfunc = self.logfunc

        content = ''
        for i in args:
            content += str(i)
        logfunc(content)
        return

    def create(self, filter_func=None):
        ''' 設定をfixしてデータセット作成を行う '''
        # writerの設定をfixする
        self.dstwriter.setup(self.srcreader, filter_func)
        
        # データセット件数を調査
        ndata = self.srcreader.datanumber()

        # 設定を表示する
        self.log('データセット作成を開始します。')
        self.log('元データのパス: ', self.srcreader.srcpath)
        self.log('元データのアーカイブ形式: ', self.srcreader.storetype.name)
        self.log('元データのデータ形式: ', self.srcreader.datatype.name)
        self.log('作成するデータセットのパス: ', self.dstwriter.dstpath)
        self.log('作成するデータセットのアーカイブ形式: ', self.dstwriter.storetype.name)
        self.log('作成するデータセットファイルの個数: ', self.dstwriter.ngroups)
        self.log('データセット名: ', self.dstwriter.basename)
        self.log('検出されたデータ件数: ', ndata)

        self.log('##############################')
        
        # データセット作成開始
        for counter in range(ndata):
            self.dstwriter.appendNext()
            if counter % 100 == 0 and counter > 0:
                self.log(counter, '件まで作成しました。')

        self.log('データセット作成が終了しました。')
        return
        
        

        
        

