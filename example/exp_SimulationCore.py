import epidag as dag

__author__ = 'TimeWz667'


scr = '''
Pcore BMI {
    b0 ~ norm(12, 1)
    b1 = 0.5 # gggg
    pf ~ beta(8, 20)
    foodstore ~ binom(100, pf)
    b0r ~ norm(0, .01)
    ageA ~ norm(20, 3)
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

bj = dag.bn_script_to_json(scr)
print(bj)
bn = dag.BayesianNetwork(bj)


hie = {
    'country': ['area'],
    'area': ['b0r', 'ps', 'foodstore', 'agA', 'agB'],
    'agA': ['bmiA', 'ageA', 'sexA'],
    'agB': ['bmiB', 'ageB']
}

sc = dag.as_simulation_core(bn, hie,
                            root='country',
                            random=['muA'],
                            out=['foodstore', 'bmiA', 'bmiB'])

pc = sc.generate('Taiwan', {'sd': 1})
pc_taipei = pc.breed('Taipei', 'area')
pc_taipei.breed('A1', 'agA')
pc_taipei.breed('A2', 'agA')
pc_taipei.breed('B1', 'agB')
b2 = pc_taipei.breed('B2', 'agB')
b2.get_sibling('B3')
pc.deep_print()

pc.impulse({'b1':100})
pc.deep_print()


area_proto = pc.get_prototype('area')
area_proto.print()
print(area_proto.get_samplers())
pc.deep_print()

pc_new = pc.clone(copy_sc=True)
pc_new.deep_print()

