from epidag.factory.arguments import *
from epidag.factory.workshop import *

__all__ = ['get_workshop']


WorkshopDict = dict()


def get_workshop(name):
    if not isinstance(name, str):
        raise NameError('A workshop name must be string')
    # todo locker
    try:
        ws = WorkshopDict[name]
    except KeyError:
        ws = Workshop()
        WorkshopDict[name] = ws
    # todo finally release locker
    return ws
