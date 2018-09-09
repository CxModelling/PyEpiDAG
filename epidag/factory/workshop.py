import inspect
from epidag.util import parse_function, ParsedFunction
from epidag.factory.arguments import ValidationError, NotNull


class Creator:
    def __init__(self, name, cls, args, meta):
        self.Name = name
        self.Class = cls
        self.Arguments = args
        self.MetaList = meta
        self.ArgList = [arg.Name for arg in self.Arguments]

    def create(self, args, meta=None):
        obj = self.Class(**args)
        if meta:
            try:
                obj.__dict__.update({k: v for k, v in meta.items() if k in self.MetaList})
            except AttributeError:
                pass
        return obj

    def validate_arguments(self, args, resources=None):
        for vld in self.Arguments:
            name = vld.Name
            try:
                value = args[name]
                vld(value, resources)
            except KeyError:
                if not vld.Optional:
                    raise ValueError('{} is not optional'.format(name))
            except ValidationError as e:
                raise e
            return True

    def correct_arguments(self, args, resources):
        """
        Adjust arguments
        :param args: arguments which have been validated
        :type args: dict
        :param resources: external resource of arguments
        :return: arguments which can be used in construction
        """
        fil = dict()
        for vld in self.Arguments:
            name = vld.Name
            try:
                value = args[name]
                fil[name] = vld.correct(value, resources)
            except ValueError:
                fil[name] = vld.Default
        return fil

    def parsed_function_to_arguments(self, pf, loc, resources):
        f_args = pf.get_arguments(loc) if loc else pf.get_arguments()
        for i, arg in enumerate(f_args):
            if 'key' not in arg:
                arg['key'] = self.Arguments[i].Name
            else:
                break

        kv = {arg['key']: arg['value'] for arg in f_args}
        args, meta = self.split_args_meta(kv)
        return self.correct_arguments(args, resources), meta

    def get_form(self, resource=None):
        return {
            'Type': self.Name,
            'Args': [arg.to_form(resource) for arg in self.Arguments]
        }

    def split_args_meta(self, kv):
        args, meta = dict(), dict()
        for k, v in kv.items():
            if k in self.ArgList:
                args[k] = v
            elif k in self.MetaList:
                meta[k] = v
        return args, meta


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

    def register(self, tp, cls, args, meta=None):
        meta = meta if meta else list()

        sig = inspect.signature(cls)
        args_map = {arg.Name: arg for arg in args}
        args = list()
        for k, v in sig.parameters.items():
            try:
                arg = args_map[k]
                if v.default is sig.empty:
                    arg.Optional = False
            except KeyError:
                if v.default is sig.empty:
                    arg = NotNull(k, opt=False)
                else:
                    arg = NotNull(k, opt=True)
            args.append(arg)
        self.Creators[tp] = Creator(tp, cls, args, meta)

    def create(self, tp, meta=None, **kwargs):
        try:
            creator = self.Creators[tp]
        except KeyError as e:
            raise e
        if meta:
            args = kwargs
        else:
            args, meta = creator.split_args_meta(kwargs)
        args = creator.correct_arguments(args, self.Resources)
        return creator.create(args, meta)

    def create_from_json(self, js):
        """
        Create an object from json form. The form must have 'Type' and 'Args'.
        The other values will be treated as meta information.
        All arguments will be validated.
        :param js: {'Type':..., 'Args':...}
        :return: object of js['Type']
        """
        tp = js['Type']
        args = {k: v for k, v in js['Args'].items()}

        try:
            creator = self.Creators[tp]
        except KeyError as e:
            raise e

        try:
            creator.validate_arguments(args, self.Resources)
        except ValidationError as e:
            raise e

        meta = {k: v for k, v in js.items() if k not in ['Args', 'Type']}
        obj = self.create(tp, meta=meta, **args)
        try:
            obj.json = js
        except AttributeError:
            pass

        return obj

    def create_from_input(self, inp):
        """
        Create an object from input form. All information should be validated.
        :param inp: {'Type':..., 'Args':...}
        :return: object of inp['Type']
        """
        tp = inp['Type']
        args = {k: v for k, v in inp['Args'].items()}
        return self.create(tp, **args)

    def parse(self, fn, loc=None):
        """
        Create an object from a string of a function
        :param fn: f(..., ..., ...)
        :param loc: local information
        :return: object of f
        """

        if not isinstance(fn, ParsedFunction):
            fn = fn.replace(' ', '')
            pf = parse_function(fn)
        else:
            pf = fn

        tp = pf.Function
        try:
            creator = self.Creators[tp]
        except KeyError as e:
            raise e

        args, meta = creator.parsed_function_to_arguments(pf, loc, self.Resources)
        try:
            creator.validate_arguments(args, self.Resources)
        except ValidationError as e:
            raise e

        obj = self.create(tp, meta=meta, **args)
        js = {
            'Type': tp,
            'Args': args
        }
        js.update(meta)
        try:
            obj.json = js
        except AttributeError:
            pass

        try:
            obj.source = fn
        except AttributeError:
            pass

        return obj

    def get_form(self, tp):
        return self.Creators[tp].get_form(self.Resources)

    def list(self):
        return list(self.Creators.keys())

    def __str__(self):
        products = list(self.Creators.keys())
        if products:
            return 'The workshop of ' + ', '.join(products)
        else:
            return 'A new workshop'


if __name__ == '__main__':
    from collections import namedtuple
    from epidag.factory.arguments import Options, PositiveInteger, Prob

    Ac = namedtuple('A', ('n', 'p', ))
    Bc = namedtuple('B', ('vs', ))

    class Ec:
        def __init__(self, x):
            self.Name = None
            self.X = x
            self.source = None
            self.json = None


    manager = Workshop()
    manager.register('A', Ac, [Prob('p'), PositiveInteger('n')])
    manager.register('B', Bc, [Options('vs', ['Z', 'X'])])
    manager.register('C', Bc, [Options('vs', 'ZX')])
    manager.register('D', Bc, [Options('vs', {'Z': 1, 'X': 2})])
    manager.register('E', Ec, [PositiveInteger('x')], ['Name'])

    manager.append_resource('ZX', ['Z', 'X'])

    print(manager.list())

    print('Test A')
    print(manager.get_form('A'))
    print(manager.create('A', p=0.2, n=5))

    print('Test B')
    print(manager.get_form('B'))
    print(manager.create_from_json({'Name': 'B1', 'Type': 'B', 'Args': {'vs': 'Z'}}))

    print('Test C')
    print(manager.get_form('C'))
    print(manager.create_from_input({'Name': 'C1', 'Type': 'C', 'Args': {'vs': 'Z'}}))

    print('Test D')
    print(manager.get_form('D'))
    print(manager.create('D', vs='Z'))

    print(manager.parse('A(5, 0.3)'))
    print(manager.parse('A(n=5, p=0.3)'))
    print(manager.parse('A(n=5, p=0.3*x)', {'x': 0.2}))

    print(manager.parse('E(5, Name="x")').json)
