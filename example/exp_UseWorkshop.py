import epidag.factory as fac
from collections import namedtuple

__author__ = 'TimeWz667'

print('Find a workshop being in charge of test objects')
manager = fac.get_workshop('Test')
print(manager)

# create pseudo classes
Ac = namedtuple('A', ('n', 'p'))
Bc = namedtuple('B', ('vs'))

# add creator in the workshop
manager.register('A', Ac, [fac.Prob('p'), fac.PositiveInteger('n')])
manager.register('B', Bc, [fac.Options('vs', ['Z', 'X'])])
manager.register('C', Bc, [fac.Options('vs', 'ZX')])
manager.register('D', Bc, [fac.Options('vs', {'Z': 1, 'X': 2})])
manager.append_resource('ZX', {'Z': 1, 'X': 2})

print('After appending creators')
print(manager)

print('\nCreate objects from json forms')
print('Test A')
# print(manager.get_form('A'))
print(manager.create_from_json({'Name': 'A1', 'Type': 'A', 'Args': {'p': 0.2, 'n': 5}}))

print('Test B')
# print(manager.get_form('B'))
print(manager.create_from_json({'Name': 'B1', 'Type': 'B', 'Args': {'vs': 'Z'}}))

print('Test C')
# print(manager.get_form('C'))
print(manager.create_from_json({'Name': 'C1', 'Type': 'C', 'Args': {'vs': 'Z'}}))

print('Test D')
# print(manager.get_form('D'))
print(manager.create_from_json({'Name': 'D1', 'Type': 'D', 'Args': {'vs': 'Z'}}))


print('\nCreate objects from function syntax')
print(manager.parse('A(5, 0.3)'))
print(manager.parse('A(n=5, p=0.3)'))
print(manager.parse('A(n=5, p=0.3*x)', {'x': 0.2}))
