
from threading import Thread
from utils.Timer import timer

# 用于多线程返回值，并记录每个线程的所用时间


class ThreadWithReturnValue(Thread):
    def __init__(self, func, args=()):
        super(ThreadWithReturnValue, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        # self.result = self.func(*self.args)
        self.spend_time, self.result = timer(self.func, args=self.args).run()

    def get_result(self):
        return self.result, self.spend_time