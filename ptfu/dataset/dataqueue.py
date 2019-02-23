''' dataqueue.py: マルチプロセス間でデータを受け渡しできるキューを記述する '''

from .. import kernel

class DataQueue:
    ''' マルチプロセス・マルチスレッド間でデータを受け渡しできるキューの実装。 '''

    def __init__(self, expected_size=-1):
        ''' キューに投入される予定の数を与えてキューを作成する 
        予定数を与えない場合は上限設定のないキューとなる '''
        #self.esizeq = Queue(1)
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
        #with self.esizelock.acquire():
            #n = self.esizeq.get()
            #self.esizeq.put(n)
        #return n
        return self.expected_size.value

    def pushednumber(self):
        ''' すでにpushされた数を返す '''
        pv = self.pushed.value
        return pv

    def poppednumber(self):
        ''' すでにpopされた数を返す '''
        #self.poplock.acquire()
        pv = self.popped.value
        #self.lock.release()
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
        #print('readAll in')
        self.poplock.acquire()
        data = []
        while self.hasnext():
            data.append(self.pop())
            #print('キューから1件popしました。')
        #print('readlAll out')
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
        #print('hasnext in')
        try:
            if self.datanumber() < 0: # 予定数がないときは常にTrue
                return True
            hasnext = (self.datanumber() - self.poppednumber() > 0)
            #print('hasnext out')
            return hasnext
        except:
            import traceback
            traceback.print_exc()


    def push(self, data):
        ''' キューにデータを追加する '''
        #print('push in')
        try:
            self.pushlock.acquire()
            #tup = (name, ndarray)
            self.q.put(data)
            self.pushed.value += 1
            self.pushlock.release()
            #print('push out self.pushed.value=', self.pushednumber())
            return
        except:
            import traceback
            traceback.print_exc()

    def pop(self):
        ''' キューから1件のデータを取り出す '''
        #print('pop in')
        self.poplock.acquire()
        #print('pop 0.5')
        if self.hasnext():
            #print('pop 1')
            data = self.q.get()
            #print('pop 2 data = ' + str(data[1].shape))
            if data is not None:
                #print('pop 3')
                try:
                    self.popped.value += 1
                except:
                    import traceback
                    traceback.print_exc()
                #print('pop 4')
                self.poplock.release()
                #print ('pop 5')
                #print('pop out popped=' + str(self.poppednumber()))
                return data
            #print('pop 6')
        #print('pop 7')
        self.poplock.release()
        #print('pop illegal out')
        return None