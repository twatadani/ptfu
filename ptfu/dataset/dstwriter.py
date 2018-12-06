# -*- coding: utf-8 -*-

''' DstWriterモジュール: データセットの書き出し処理を記述する '''

import os.path
from io import BytesIO
from zipfile import ZipFile
from tarfile import TarFile, TarInfo

import numpy as np

class DstWriter:
    ''' 各種データセットの書き出し処理を行うクラス '''

    def __init__(self, dststoretype, dstpath, basename):
        ''' DstWriterのイニシャライザ
        dststoretype: StoreType enumのメンバを指定する
        dstpath: データセットの親ディレクトリ名
        basename: データセットの名前 (これに分割番号が付く) '''

        self.storetype = dststoretype
        self.dstpath = os.path.expanduser(dstpath)
        self.basename = basename

        self.writers = []

        self.ngroups = 1 # 分割書き込みの際のグループ数
        self.n_per_group = -1 #分割書き込みの際の1グループあたりのデータ数

        self.srcreader = None
        self.iterator = None
        self.fixed = False # 設定を固定するロック
        self.wcount = None # 書き込みを行った件数
        return

    def setup(self, srcreader, filterfunc=None):
        ''' 設定をfixして書き込み準備完了の状態にする
        filterfunc: srcreaderから読み込んだndarrayを変換するフィルター関数。返り値もndarray '''

        self.fixed = True
        self.srcreader = srcreader
        if filterfunc is not None:
            self.filterfunc = filterfunc
        self.iterator = self.srcreader.iterator()
        ndata = srcreader.datanumber()
        self.wcount = 0
 
        # モードの判定
        if self.n_per_group < 0: # ngroupsを使うモード。
        # このモードでは最初からグループ数が決まっており、書き込みは
        # 1番目-> グループ1, 2番目-> グループ2, 3番目-> グループ3, のように
        # 1件書き込む毎に次のグループに変わる。
            self.n_per_group = -1 #何もしないのと同様
        else: # n_per_groupを使うモード
        # このモードは1グループ当たりの件数が決まっており、書き込みは
        # 1番目 -> グループ1, 2番目 -> グループ1, ... n番目 -> グループ1
        # n+1番目 -> グループ2, ...のように1つのグループのキャパシティが尽きるまでは
        # 1つのグループに書き込みを続ける。
            self.ngroups = ndata // self.n_per_group
            if ndata % self.n_per_group > 0:
                self.ngroups += 1

        # いずれのモードでもwritersを準備する
        for i in range(self.ngroups):
            dstpathi = self._create_dstpath_i(i+1)
            writer = self.storetype.writer()(dstpathi)
            self.writers.append(writer)

        return

    def _create_dstpath_i(self, i):
        ''' 内部用の関数 i番目のdstpathの文字列を作成する '''
        lastname = self.basename
        if self.ngroups == 1:
            lastname += '' + '.' + self.storetype.getext()
        else:
            lastname += '-' + str(i) + '.' + self.storetype.getext()
        return os.path.join(self.dstpath, lastname)

    def setSplitByGroups(self, n):
        ''' 書き込みのグループ数を指定して分割書き込みを行う '''
        assert n >= 1
        if self.fixed:
            raise ValueError('Split change while writing is prohibited.')
        self.ngroups = n
        self.n_per_group = -1
        return

    def setSplitByGroupNumber(self, n):
        ''' 書き込みの分割を1グループあたりのデータ数で指定する '''
        assert n >= 1
        if self.fixed:
            raise ValueError('Split change while writing is prohibited.')
        self.n_per_group = n
        self.ngroups = -1
        return

    def appendNext(self):
        ''' iteratorから得た1件のデータを書き込む '''
        if self.fixed == False:
            raise ValueError('Dstwriter is not fixes when appendNext called.')

        name, ndarray = next(self.iterator)
        if self.filterfunc is not None:
            ndarray = self.filterfunc(ndarray)
        nextwriter = self._nextwriter()
        nextwriter.appendNext(name, ndarray)
        self._increment_nextwriter()
        return

    def _nextwriter(self):
        ''' 内部用の関数 次に書き込むStoreTypeWriterを返す '''
        
        nextwriter = None
        if self.n_per_group < 0: # ngroupsを使うモード
        # このモードでは最初からグループ数が決まっており、書き込みは
        # 1番目-> グループ1, 2番目-> グループ2, 3番目-> グループ3, のように
        # 1件書き込む毎に次のグループに変わる。
            nextwriter = self.writers[self.wcount % self.ngroups]
        else: # n_per_groupを使うモード
        # このモードは1グループ当たりの件数が決まっており、書き込みは
        # 1番目 -> グループ1, 2番目 -> グループ1, ... n番目 -> グループ1
        # n+1番目 -> グループ2, ...のように1つのグループのキャパシティが尽きるまでは
        # 1つのグループに書き込みを続ける。
            nextwriter = self.writers[self.wcount // self.n_per_group]
        return nextwriter

    def _increment_nextwriter(self):
        ''' 内部用の関数 1件書き込みが終わった後、次のwriterを指定する '''
        self.wcount += 1
        return

class StoreTypeWriter:
    ''' データセット格納形式StoreTypeに対応したWriterの基底クラス '''

    def __init__(self, dstpath):
        ''' StoreTypeWriterのイニシャライザ
        dstpath: 書き込みを行うディレクトリまたはアーカイブのパス '''
        self.dstpath = dstpath # 書き込みを行うディレクトリまたはアーカイブのパス
        self.writestarted = False
        return

    def __del__(self):
        ''' StoreTypeWriterのデストラクタ
        デフォルト動作ではclose_dstのみを行う '''
        self.close_dst()

    def open_dst(self):
        ''' アーカイブのオープン デフォルトではなにもしない 
        具象クラスで必要ならばオーバーライドする '''
        return

    def close_dst(self):
        ''' アーカイブのクローズ デフォルトではなにもしない
        具象クラスで必要ならばオーバーライドする '''
        return

    def appendNext(self, name, ndarray):
        ''' iteratorから得た1件のデータを書き込む '''
        if self.writestarted == False:
            self.open_dst()
            self.writestarted = True
        self._appendNext(name, ndarray)
        return

    def _appendNext(self, name, ndarray):
        ''' iteratorから得た1件のデータを書き込む。具象クラスで定義する '''
        raise NotImplementedError

class DirWriter(StoreTypeWriter):
    ''' 生のディレクトリ内にファイルを格納するWriter '''

    def __init__(self, dstpath):
        ''' DirWriterのイニシャライザ
        dstpath: 書き込みを行うディレクトリ '''
        super(DirWriter, self).__init__(dstpath)

    def open_dst(self):
        ''' アーカイブのオープン DirWriterではディレクトリの作成を行う '''
        os.makedirs(self.dstpath, exist_ok=True)
        return

    def _appendNext(self, name, ndarray):
        ''' iteratorから得た1件のデータを書き込む '''
        fullname = os.path.join(self.dstpath, name + '.npy')
        np.save(fullname, ndarray, allow_pickle=False)
        return

class TarWriter(StoreTypeWriter):
    ''' Tarアーカイブ内にファイルを格納するWriter '''

    def __init__(self, dstpath):
        ''' TarWriterのイニシャライザ
        dstpath: 書き込みを行うディレクトリ '''
        super(TarWriter, self).__init__(dstpath)
        self.tfp = None # TarFile object
        # 名前が.tarの場合.tar.gzに変更する
        if os.path.splitext(dstpath)[-1] == '.tar':
            self.dstpath += '.gz'

    def open_dst(self):
        ''' アーカイブのオープン TarWriterではTarFileのオープンを行う '''
        if self.tfp is None:
            parent_dir = os.path.dirname(self.dstpath)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            self.tfp = TarFile.open(name=self.dstpath, mode='w:gz')
        return

    def close_dst(self):
        ''' アーカイブのクローズ TarWriterではTarFileのクローズを行う '''
        if self.tfp is not None:
            self.tfp.close()
            self.tfp = None

    def _appendNext(self, name, ndarray):
        ''' iteratorから得た1件のデータを書き込む '''
        buf = BytesIO()
        np.save(buf, ndarray, allow_pickle=False)
        info = TarInfo(name=name + '.npy')
        info.size = len(buf.getbuffer())
        buf.seek(0)
        self.tfp.addfile(info, fileobj=buf)
        buf.close()
        return

class ZipWriter(StoreTypeWriter):
    ''' 生のディレクトリ内にファイルを格納するWriter '''

    def __init__(self, dstpath):
        ''' ZipWriterのイニシャライザ
        dstpath: 書き込みを行うディレクトリ '''
        super(ZipWriter, self).__init__(dstpath)
        self.zfp = None # ZipFile object

    def open_dst(self):
        ''' アーカイブのオープン ZipWriterではzipアーカイブのオープンを行う '''
        print('ZipWriter open_dst called.')
        if self.zfp is None:
            parent_dir = os.path.dirname(self.dstpath)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            self.zfp = ZipFile(self.dstpath, mode='w')
        return

    def close_dst(self):
        ''' アーカイブのクローズ ZipWriterではzipアーカイブのクローズを行う '''
        print('ZipWriter close_dst called.')
        if self.zfp is not None:
            self.zfp.close()
            self.zfp = None

    def _appendNext(self, name, ndarray):
        ''' iteratorから得た1件のデータを書き込む '''
        buf = BytesIO()
        np.save(buf, ndarray, allow_pickle=False)
        print('self.writestarted=', self.writestarted)
        print('self.zfp: ', self.zfp)
        self.zfp.writestr(name + '.npy', buf.getbuffer())
        buf.close()
        return

class TFRecordWriter(StoreTypeWriter):
    ''' TFRecord形式のデータセット格納を行うWriter '''

    def __init__(self, dstpath):
        ''' TFRecordWriterのイニシャライザ
        dstpath: 書き込みを行うディレクトリ '''
        super(TFRecordWriter, self).__init__(dstpath)
        self.writer = None # TFRecordWriter object

    def open_dst(self):
        ''' アーカイブのオープン TFRecordWriterではTFRecordWriterオブジェクトを生成する '''
        import tensorflow as tf
        parent_dir = os.path.dirname(self.dstpath)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        if self.writer is None:
            option = tf.python_io.TFRecordOptions(
                compression_type = tf.python_io.TFRecordCompressionType.GZIP)
            self.writer = tf.python_io.TFRecordWriter(self.dstpath, options=option)
        return

    def close_dst(self):
        ''' アーカイブのクローズ TFRecordWriterではTFRecordWriterをcloseする '''
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _appendNext(self, name, ndarray):
        ''' iteratorから得た1件のデータを書き込む '''
        import tensorflow as tf
        features = tf.train.Features(
            feature= {
                'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode()])),
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ndarray.tobytes()]))
            })
        example = tf.train.Example(features=features)
        self.writer.write(example.SerializeToString())
        return
