from .util import parse_function
from .arguments import ValidationError

__author__ = 'TimeWz667'
__all__ = ['getWorkshop']


class Creator:
    def __init__(self, name, cls, args):
        self.Name = name
        self.Class = cls
        self.Arguments = args

    def get_form(self, resource=None):
        return {
            'Name': self.Name,
            'Args': [arg.to_form(resource) for arg in self.Arguments]
        }

    def create(self, name, args):
        return self.Class(name, **args)

    def validate_arguments(self, args, resource=None):
        for vld in self.Arguments:
            name = vld.Name
            try:
                value = args[name]
                vld(value, resource)
            except KeyError:
                if not vld.Optional:
                    return False
            except ValidationError:
                return False
            return True

    def align_arguments(self, args, resource=None):
        """

        :param args: arguments which have been validated
        :param resource: external resource of arguments
        :return: arguments which can be used in construction
        """
        fil = dict()
        kwargs = {arg[0]: arg[1] for arg in args if isinstance(arg, tuple)}
        args = [arg for arg in args if not isinstance(arg, tuple)]
        for arg, vld in zip(args, self.Arguments):
            try:
                vld(arg, resource)
            except ValidationError:
                return False
            except KeyError:
                if not vld.Optional:
                    return False
                else:
                    continue
            fil[vld.Name] = arg
        for vld in self.Arguments[len(args):]:
            name = vld.Name
            try:
                value = kwargs[name]
                vld(value, resource)
            except ValidationError:
                return False
            except KeyError:
                if not vld.Optional:
                    return False
                else:
                    continue
            fil[name] = value
        return fil

    def reform_arguments(self, args, resource=None):
        """

        :param args: arguments which have been validated
        :param resource: external resource of arguments
        :return: arguments which can be used in construction
        """
        fil = dict()
        for vld in self.Arguments:
            name = vld.Name
            try:
                value = args[name]
                fil[name] = vld.correct(value, resource)
            except KeyError:
                if not vld.Optional:
                    raise ValueError
                else:
                    continue
        return fil


class Workshop:
    def __init__(self):
        self.Resources = dict()
        self.Creators = dict()

    def append_resource(self, name, res):
        self.Resources[name] = res

    def renew_resources(self, new):
        self.Resources = dict(new)

    def clear_resources(self):
        self.Resources = dict()

    def register(self, tp, cls, args):
        self.Creators[tp] = Creator(tp, cls, args)

    def validate(self, js, logger=None):
        """

        :param js: definition of object in json form
        :param logger: logging records
        :return: true if validated
        """
        args = js['Args']
        creator = self.Creators[js['Type']]
        if creator.validate_arguments(args, self.Resources):
            return True
        else:
            if logger:
                logger.debug('Invalidated arguments')
            return False

    def create(self, js, logger=None):
        try:
            name = js['Name']
            args = js['Args']
            creator = self.Creators[js['Type']]
            args = creator.reform_arguments(args, self.Resources)
            res = creator.create(name, args)
            try:
                res.json = js
            except AttributeError:
                pass
            return res
        except TypeError:
            if logger:
                logger.debug('Object creation failed')
            return

    def from_json(self, js, logger=None):
        name = js['Name']
        args = js['Args']
        res = self.Creators[js['Type']].create(name, args)
        res.json = js
        return res

    def parse(self, name, fn=None, env=None, js=True):
        if not fn:
            fn = name
        f, pars = parse_function(fn, env)
        creator = self.Creators[f]
        pars = creator.align_arguments(pars, self.Resources)
        res = creator.create(name, pars)
        if js:
            try:
                res.json = {'Name': name, 'Args': pars}
            except AttributeError:
                pass
        return res

    def get_form(self, tp):
        return self.Creators[tp].get_form(self.Resources)

    def list(self):
        return list(self.Creators.keys())


WorkshopDict = dict()


def getWorkshop(name):
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


if __name__ == '__main__':
    from collections import namedtuple
    from factory.arguments import Options, PositiveInteger, Prob

    Ac = namedtuple('A', ('Name', 'n', 'p'))
    Bc = namedtuple('B', ('Name', 'vs'))

    manager = Workshop()
    manager.register('A', Ac, [Prob('p'), PositiveInteger('n')])
    manager.register('B', Bc, [Options('vs', ['Z', 'X'])])
    manager.register('C', Bc, [Options('vs', 'ZX')])
    manager.register('D', Bc, [Options('vs', {'Z': 1, 'X': 2})])
    manager.append_resource('ZX', ['Z', 'X'])

    print(manager.list())

    print('Test A')
    print(manager.get_form('A'))
    print(manager.create({'Name': 'A1', 'Type': 'A', 'Args': {'p': 0.2, 'n': 5}}))

    print('Test B')
    print(manager.get_form('B'))
    print(manager.create({'Name': 'B1', 'Type': 'B', 'Args': {'vs': 'Z'}}))

    print('Test C')
    print(manager.get_form('C'))
    print(manager.create({'Name': 'C1', 'Type': 'C', 'Args': {'vs': 'Z'}}))

    print('Test D')
    print(manager.get_form('D'))
    print(manager.create({'Name': 'D1', 'Type': 'D', 'Args': {'vs': 'Z'}}))

    print(manager.parse('A2', 'A(0.3, 5)'))
    print(manager.parse('A3', 'A(n=5, p=0.3)'))
    print(manager.parse('A4', 'A(n=5, p=0.3*x)', {'x': 0.2}))
