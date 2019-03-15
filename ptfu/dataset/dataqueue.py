''' dataqueue.py: マルチプロセス間でデータを受け渡しできるキューを記述する '''

from .. import kernel

class DataQueue:
    ''' マルチプロセス・マルチスレッド間でデータを受け渡しできるキューの実装。 '''

    def __init__(self, expected_size=-1):
        ''' キューに投入される予定の数を与えてキューを作成する 
        予定数を与えない場合は上限設定のないキューとなる '''
        self.esizelock = kernel.manager.RLock()
        self.expected_size = kernel.manager.Value('i', expected_size)
        self.popped = kernel.manager.Value('i', 0)
        self.pushed = kernel.manager.Value('i', 0)
        self.q = kernel.manager.Queue()
        self.pushlock = kernel.manager.RLock()
        self.poplock = kernel.manager.RLock()
        return

    def datanumber(self):
        ''' 予定されたデータ数を返す '''
        return self.expected_size.value

    def pushednumber(self):
        ''' すでにpushされた数を返す '''
        pv = self.pushed.value
        return pv

    def poppednumber(self):
        ''' すでにpopされた数を返す '''
        pv = self.popped.value
        return pv

    def qsize(self):
        ''' 現在のおおよそのキューサイズを得る '''
        self.poplock.acquire()
        self.pushlock.acquire()
        qsize = self.q.qsize()
        self.pushlock.release()
        self.poplock.release()
        return qsize

    def readAll(self):
        ''' すべてのデータを読み取る。途中でキューが空になった場合は補充されるまで待ちながら読み取る '''
        self.poplock.acquire()
        data = []
        while self.hasnext():
            data.append(self.pop())
        self.poplock.release()
        return data

    def putAll(self, collection):
        ''' collectionで与えられたデータをすべてキューに入れる。 '''
        self.pushlock.acquire()
        list(map(lambda x: self.push(x), collection))
        self.pushlock.release()
        return

    def hasnext(self):
        ''' キューにまだ取り出せるデータの予定があるかどうかを調べる。 '''
        try:
            if self.datanumber() < 0: # 予定数がないときは常にTrue
                return True
            hasnext = (self.datanumber() - self.poppednumber() > 0)
            return hasnext
        except:
            import traceback
            traceback.print_exc()


    def push(self, data):
        ''' キューにデータを追加する '''
        lock_acquired = False
        try:
            self.pushlock.acquire()
            lock_acquired = True
            self.q.put(data)
            self.pushed.value += 1
            self.pushlock.release()
            lock_acquired = False
            return
        except:
            if lock_acquired:
                self.pushlock.release()
                lock_acquired = False
            import traceback
            traceback.print_exc()

    def pop(self):
        ''' キューから1件のデータを取り出す '''
        self.poplock.acquire()
        if self.hasnext():
            data = self.q.get()
            if data is not None:
                try:
                    self.popped.value += 1
                except:
                    import traceback
                    traceback.print_exc()
                self.poplock.release()
                return data
        self.poplock.release()
        return None
