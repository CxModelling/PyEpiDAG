from logging import getLogger, FileHandler, StreamHandler
import pandas as pd

__author__ = 'TimeWz667'
__all__ = ['Monitor']


class Monitor:
    def __init__(self, name):
        self.Title = name
        self.Logger = getLogger(name)
        self.Logger.addHandler(StreamHandler())
        self.Records = []
        self.Time = 0
        self.Last = {"Time": self.Time}

    def info(self, msg, *arg, **kwargs):
        self.Logger.info(msg, *arg, **kwargs)

    def warning(self, msg, *arg, **kwargs):
        self.Logger.warning(msg, *arg, **kwargs)

    def error(self, msg, *arg, **kwargs):
        self.Logger.error(msg, *arg, **kwargs)

    def set_log_path(self, filename):
        self.Logger.addHandler(FileHandler(filename))

    def initialise(self, time=None):
        self.Time = time if time else 0
        self.Records.clear()
        self.Last = {"Time": time}

    def step(self, time=None):
        time = time if time else self.Time + 1
        if time < self.Last["Time"]:
            self.Time = self.Last["Time"] + 1
        else:
            self.Time = time
        self.Records.append(self.Last)
        self.Last = {"Time": time}

    def keep(self, **kwargs):
        self.Last.update(kwargs)

    @property
    def Trajectories(self):
        dat = pd.DataFrame(self.Records)
        return dat.set_index("Time")

    def save_trajectories(self, filename):
        self.Trajectories.to_csv(filename)


if __name__ == '__main__':
    mon = Monitor("Test")
    mon.keep(Size=4)
    mon.step()
    mon.keep(Size=6)
    mon.step()
    print(mon.Trajectories)
