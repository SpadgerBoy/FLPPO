
import time

# 用于记录一个函数的运行时间，并返回该函数的返回值


class timer(object):
    def __init__(self, func, args=()):
        super(timer, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        time1 = time.time()
        res = self.func(*self.args)
        time2 = time.time()
        spend_time = time2 - time1
        return spend_time, res