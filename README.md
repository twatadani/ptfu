# Personal TensorFlow Utility (ptfu) ver 0.2

Python / Tensorflowで開発を行うための個人的なユーティリティー群

## 実装履歴

* version 0.1 - プロジェクトを作成。__init__.pyを作成しパッケージとしてデザイン。SmartSessionおよびTFConfigを実装。
* version 0.2 - datasetサブパッケージを作成

## ptfuの全体構成

ptfu自体を一つのpythonパッケージとすることで
```python
import ptfu
```
の形式で呼び出せるようにする。


# ptfu 簡易マニュアル

## データセットの作成

datasetサブパッケージを利用する。

### シンプルなサンプルプログラム

```python
from ptfu.dataset import DataSetCreator, DataType, StoreType

# DatasetCreatorオブジェクトを生成する

# srcdatatype: 元データの画像形式 DataType enumから選択する
srcdatatype = DataType.JPG
# srcstoretype: 元データの格納形式 StoreType enumから選択する
srcstoretype = StoreType.ZIP
# srcpath: 元データの保存場所。DIRの場合はディレクトリ、それ以外の場合はアーカイブファイルを指定する
srcpath = '/foo/bar/srcdata.zip'

# dststoretype: 作成するデータセットの格納形式 StoreType enumから選択する
dststoretype = StoreType.TFRECORD
# dstpath: 作成するデータセットの保存場所。ディレクトリで指定する
dstpath = '/directory/to/be/stored'
# datasetname 作成するデータセットの名前
datasetname = 'mydataset'

creator = DatasetCreator(srcdatatype, srcstoretype, srcpath, dststoretype, dstpath, datasetname)

# データセット作成
creator.create()
```

