from abc import ABCMeta, abstractmethod
import re
import os

__author__ = 'TimeWz667'


__all__ = ['ValidationError',
           'Float', 'PositiveFloat', 'NegativeFloat', 'Prob',
           'Integer', 'PositiveInteger', 'NegativeInteger',
           'String', 'RegExp', 'Path', 'List', 'ProbTab',
           'Options', 'NotNull']


class ValidationError(ValueError):
    def __init__(self, message='', *args):
        """
        Error message when validation fails
        :param message:
        :param args:
        """
        ValueError.__init__(self, message, *args)


class Argument(metaclass=ABCMeta):
    def __init__(self, name, tp, des, opt):
        """
        Abstract object of all arguments
        :param name: name
        :param tp: data type
        :param des: description
        :param opt: optional or not
        """
        self.Name = name
        self.Description = des if des else name
        self.Type = tp
        self.Optional = bool(opt)
        self.Default = None

    def __call__(self, value, resource=None):
        return self.check(value, resource)

    def to_form(self, resource=None):
        """
        Render the form of the argument
        :param resource:
        :return:
        """
        return {
            'Name': self.Name,
            'Type': self.Type,
            'Description': self.Description,
            'Optional': self.Optional
        }

    @abstractmethod
    def check(self, value, resource=None):
        """
        Check if the value is legal or not
        :param value: input value
        :param resource:
        :return: true if the value is legal
        """
        pass

    def correct(self, value, resource=None):
        """
        Render legal value as possible
        :param value: input value
        :param resource:
        :return: decorated value
        """
        try:
            return resource[value]
        except TypeError:
            return value
        except KeyError:
            return value


class NotNull(Argument):
    def __init__(self, name, des=None, opt=False):
        Argument.__init__(self, name, 'notnull', des, opt)

    def check(self, value, resource=None):
        if value:
            return True
        else:
            raise ValidationError('Missing value')


class Float(Argument):
    def __init__(self, name, lower=float('-inf'), upper=float('inf'), default=0.0, des=None, opt=False):
        Argument.__init__(self, name, 'float', des, opt)
        self.Lower = lower
        self.Upper = upper
        self.Default = default

    def check(self, value, resource=None):
        try:
            value = float(value)
        except ValueError:
            raise ValidationError('Invalid value type')
        except TypeError:
            raise ValidationError('Missing value')

        if self.Lower > value:
            raise ValidationError('The value is below lower bond')
        elif self.Upper < value:
            raise ValidationError('The value is beyond upper bond')
        return value

    def to_form(self, resource=None):
        js = Argument.to_form(self, resource)
        js.update({'Lower': self.Lower,
                   'Upper': self.Upper,
                   'Default': self.Default})
        return js

    def correct(self, value, resource=None):
        value = Argument.correct(self, value, resource)
        return float(value)


class PositiveFloat(Float):
    def __init__(self, name, default=1.0, des=None, opt=False):
        Float.__init__(self, name, lower=0.0, default=default, des=des, opt=opt)


class NegativeFloat(Float):
    def __init__(self, name, default=1.0, des=None, opt=False):
        Float.__init__(self, name, upper=0.0, default=default, des=des, opt=opt)


class Prob(Float):
    def __init__(self, name, default=0.5, des=None, opt=False):
        Float.__init__(self, name, lower=0.0, upper=1.0, default=default, des=des, opt=opt)


class Integer(Argument):
    def __init__(self, name, lower=float('-inf'), upper=float('inf'), default=0, des=None, opt=False):
        Argument.__init__(self, name, 'int', des, opt)
        self.Lower = lower
        self.Upper = upper
        self.Default = default

    def check(self, value, resource=None):
        try:
            value = int(value)
        except ValueError:
            raise ValidationError('Invalid value type')
        except TypeError:
            raise ValidationError('Missing value')

        if self.Lower >= value:
            raise ValidationError('The value is below lower bond')
        elif self.Upper <= value:
            raise ValidationError('The value is beyond upper bond')
        return True

    def to_form(self, resource=None):
        js = Argument.to_form(self, resource)
        js.update({'Lower': self.Lower,
                   'Upper': self.Upper,
                   'Default': self.Default})
        return js

    def correct(self, value, resource=None):
        value = Argument.correct(self, value, resource)
        return int(value)


class PositiveInteger(Integer):
    def __init__(self, name, default=1, des=None, opt=False):
        Integer.__init__(self, name, lower=0, default=default, des=des, opt=opt)


class NegativeInteger(Integer):
    def __init__(self, name, default=1, des=None, opt=False):
        Integer.__init__(self, name, upper=0, default=default, des=des, opt=opt)


class String(Argument):
    def __init__(self, name, default=None, des=None, opt=False):
        Argument.__init__(self, name, 'str', des, opt)
        self.Default = default

    def check(self, value, resource=None):
        if not isinstance(value, str):
            raise ValidationError('Value is not a string')
        return True

    def to_form(self, resource=None):
        js = Argument.to_form(self, resource)
        js.update({'Default': self.Default})
        return js


class RegExp(String):
    def __init__(self, name, reg, flags=re.I, default=None, des=None, opt=False):
        String.__init__(self, name, default, des=des, opt=opt)
        self.RF = reg, str(re.I)
        self.RE = re.compile(reg, flags)

    def check(self, value, resource=None):
        match = self.RE.match(value)
        if not match:
            raise ValidationError('Value parse failed')
        return True

    def to_form(self, resource=None):
        js = Argument.to_form(self, resource)
        js.update({
                'RegExp': self.RF[0],
                'Flag': self.RF[1],
                'Default': self.Default})
        return js


class Path(String):
    def __init__(self, name, default=None, des=None, opt=False):
        String.__init__(self, name, default, des=des, opt=opt)

    def check(self, value, resource=None):
        os.path.isfile(value)

    def to_form(self, resource=None):
        js = Argument.to_form(self, resource)
        js.update({'Default': self.Default})
        return js


class List(Argument):
    def __init__(self, name, size=None, des=None, opt=False):
        Argument.__init__(self, name, 'list', des=des, opt=opt)
        self.Size = size

    def check(self, value, resource=None):
        try:  # todo list input
            value = eval(value, resource)
        except NameError:
            raise ValidationError('Invalidated value')

        if isinstance(value, list):
            raise ValidationError('Invalidated value type')

        if self.Size:
            if len(value) is not self.Size:
                raise ValidationError('Unfitted value length')
        return True

    def correct(self, value, resource=None):
        value = Argument.correct(self, value, resource)
        if isinstance(value, list):
            return value
        else:
            return eval(value, resource)


class Options(Argument):
    def __init__(self, name, options, des=None, opt=False):
        Argument.__init__(self, name, 'option', des=des, opt=opt)
        if isinstance(options, list):
            self.Options = [str(o) for o in options]
        elif isinstance(options, dict):
            self.Options = options
        else:
            self.Options = str(options)

    def check(self, value, resource=None):
        opts = resource[self.Options] if isinstance(self.Options, str) else self.Options

        if isinstance(opts, list):
            if value not in opts:
                raise ValidationError('Invalid value')
            return True
        elif isinstance(opts, dict):
            if value in opts or value in opts.values():
                return True
            else:
                raise ValidationError('Invalid value')
        else:
            raise ValidationError('Invalid value type')

    def to_form(self, resource=None):
        if isinstance(self.Options, str):
            opts = resource[self.Options]
        else:
            opts = self.Options

        if isinstance(opts, dict):
            opts = list(opts.keys())

        js = Argument.to_form(self, resource)
        js['Options'] = opts
        return js

    def correct(self, value, resource=None):
        opts = resource[self.Options] if isinstance(self.Options, str) else self.Options

        if isinstance(opts, list):
            return value
        elif isinstance(opts, dict):
            return opts[value]
        return None


class ProbTab(Argument):
    def __init__(self, name, des=None, opt=False):
        Argument.__init__(self, name, 'probtab', des=des, opt=opt)

    def check(self, value, resource=None):
        if isinstance(value, str) and value in resource:
            value = resource[value]
        if isinstance(value, str):
            try:
                value = eval(value, resource)
            except NameError:
                raise ValidationError('Invalid syntax')

        if not isinstance(value, dict):
            raise ValidationError('Invalid value type')

        for v in value.values():
            try:
                float(v)
            except ValueError:
                raise ValidationError('Invalid probability')

        return True

    def correct(self, value, resource=None):
        value = Argument.correct(self, value, resource)
        if isinstance(value, str):
            value = eval(value, resource)

        return {str(k): max(float(v), 0) for k, v in value.items()}


if __name__ == '__main__':
    def try_arg(arg, truthy, falsy, res=None):
        print(arg.to_form(res))

        try:
            arg(truthy, res)
            print('Good')
        except ValidationError:
            print('Bad')

        try:
            arg(falsy, res)
            print('Bad')
        except ValidationError:
            print('Good')

    try_arg(PositiveFloat('+1.0'), 1, -1)

    try_arg(NegativeInteger('-1'), -1, 1)

    try_arg(RegExp('A-B', r'\w+-\w+'), 'A-B', 'A+B')

    try_arg(Options('A, B, C', ['A', 'B', 'C']), 'A', 'D')

    try_arg(Options('A, B, C', 'ABC'), 'A', 'D', res={'ABC': ['A', 'B', 'C']})

    from collections import namedtuple
    entry = namedtuple('entry', 'Name')

    try_arg(Options('A, B, C', 'ABC'), 'A', 'D', res={'ABC': {'A': entry('A'), 'B': entry('B'), 'C': entry('C')}})

    try_arg(ProbTab('k,v'), 'ABC', 'D', res={'ABC': {'A': 0.2, 'B': 0.3, 'C': 0.5}})

    try_arg(ProbTab('k,v'), {'A': 0.2, 'B': 0.3, 'C': 0.5}, {'A': 'A', 'B': 0.3, 'C': 0.5})
