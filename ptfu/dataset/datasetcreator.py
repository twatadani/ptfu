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

        from concurrent.futures import ThreadPoolExecutor

        self.srcreader = None
        self.dstwriter = None

        if srcdatatype is not None and srcstoretype is not None and srcpath is not None:
            self.srcreader = SrcReader(srcdatatype, srcstoretype, srcpath)

        if dststoretype is not None and dstpath is not None and datasetname is not None:
            self.dstwriter = DstWriter(dststoretype, dstpath, datasetname)

        #self.logfunc = None

        self.executor = ThreadPoolExecutor()
        return

    def setSplitByGroups(self, n):
        ''' 書き込みのグループ数を指定して分割書き込みを行う '''
        self.dstwriter.setSplitByGroups(n)
        return

    def setSplitByGroupNumber(self, n):
        ''' 書き込みの分割を1グループあたりのデータ数で指定する '''
        self.dstwriter.setSplitByGroupNumber(n)
        return

    def create(self, filter_func=None):
        ''' 設定をfixしてデータセット作成を行う '''
        from concurrent.futures import wait
        from ..logger import get_default_logger
        logger = get_default_logger()

        # writerの設定をfixする
        self.dstwriter.setup(self.srcreader, filter_func)
        
        # データセット件数を調査
        ndata = self.srcreader.datanumber()

        # 設定を表示する
        logger.log('データセット作成を開始します。')
        logger.log('元データのパス: ' + str(self.srcreader.srcpath))
        logger.log('元データのアーカイブ形式: ' + self.srcreader.storetype.name)
        logger.log('元データのデータ形式: ' + self.srcreader.datatype.name)
        logger.log('作成するデータセットのパス: ' + self.dstwriter.dstpath)
        logger.log('作成するデータセットのアーカイブ形式: ' +  self.dstwriter.storetype.name)
        logger.log('作成するデータセットファイルの個数: ' + str(self.dstwriter.ngroups))
        logger.log('データセット名: ' + self.dstwriter.basename)
        logger.log('検出されたデータ件数: ' + str(ndata))

        logger.log('##############################')
        
        # データセット作成開始

        nwritten = 0
        writebatch = 100
        while nwritten < ndata:
            nextnwrite = min(ndata-nwritten, writebatch)
            futures = []
            for _ in range(nextnwrite):
                futures.append(self.executor.submit(self.dstwriter.appendNext))

            wait(futures)
            for future in futures:
                future.result(1) # 例外が発生していればここで送出される
            nwritten += nextnwrite
            logger.log(str(nwritten) +  '件まで作成しました。')

        logger.log('データセット作成が終了しました。')
        return
