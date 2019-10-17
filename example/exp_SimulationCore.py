import epidag as dag

__author__ = 'TimeWz667'


scr = '''
Pcore BMI {
    b0 ~ norm(12, 1)
    b1 = 0.5 # gggg
    pf ~ beta(8, 20)
    foodstore ~ binom(100, pf)
    b0r ~ norm(0, .01)
    ageA # ~ norm(20, 3)
    ageB ~ norm(30, 2)
    ps ~ beta(5, 6)
    sexA ~ cat({'m': ps, 'f': 1-ps})
    muA = b0 + b0r + b1*ageA
    bmiA ~ norm(muA, sd)
    sdB = sd * 0.5
    muB = b0 + b0r + b1*ageB
    bmiB ~ norm(muB, sdB)
}
'''

bn = dag.bayes_net_from_script(scr)


root = dag.NodeSet('country')
node_area = root.new_child('area', as_fixed=['b0r', 'ps'], as_floating=['foodstore'])
node_area.new_child('agA', as_fixed=['ageA', 'sexA'], as_floating=['bmiA'])
node_area.new_child('agB', as_fixed=['ageB'], as_floating=['bmiB'])

sc = dag.as_simulation_core(bn, root, True)

root.print()
root.print_samplers()

pc = sc.generate('Taiwan', {'sd': 1})
print(pc.list_samplers())
pc_taipei = pc.breed('Taipei', 'area')
pc_taipei.breed('A1', 'agA', {'ageA': 5})
pc_taipei.breed('A2', 'agA', {'ageA': 4})
pc_taipei.breed('B1', 'agB')
print(pc_taipei.list_samplers())
b2 = pc_taipei.breed('B2', 'agB')
b2.get_sibling('B3')
pc.deep_print()
print(b2.list_samplers())

sam_bmiB = b2.get_sampler('bmiB')

print(sam_bmiB.sample(5))

pc.impulse({'b1': 100})
pc.deep_print()


area_proto = pc.get_prototype('area')
area_proto.print()
print(area_proto.get_samplers())
pc.deep_print()

pc_new = pc.clone(copy_sc=True)
pc_new.deep_print()

