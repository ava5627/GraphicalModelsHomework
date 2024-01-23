import time
from multiprocessing import Queue


class OutputManager:
    def __init__(self, names):
        self.queues = {}
        self.messages = {}
        for name in names:
            self.queues[name] = Queue()
            self.messages[name] = ""
        self.terminated = False

    def run(self):
        while not self.terminated:
            self.print()
        print(f"\033[{len(self.messages)}B", end="")

    def print(self):
        for name, q in self.queues.items():
            if not q.empty():
                self.messages[name] = q.get()
            print(name, self.messages[name], "\033[K")
        print(f"\033[{len(self.messages)}A", end="")
        time.sleep(1)

    def terminate(self):
        self.terminated = True
