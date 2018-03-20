import epidag as dag

__author__ = 'TimeWz667'


scr = '''
Pcore BMI {
    b0 ~ norm(12, 1)
    b1 = 0.5
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
bn = dag.BayesianNetwork(bj)


hie = {
    'country': ['area'],
    'area': ['b0r', 'ps', 'foodstore', 'agA', 'agB'],
    'agA': ['bmiA', 'ageA', 'sexA'],
    'agB': ['bmiB', 'ageB']
}


# ng = dag.form_hierarchy(bn, hie, root='country')
# bp = dag.formulate_blueprint(bn, ng, random=['muA'], out=['foodstore', 'bmiA', 'bmiB'])
sc = dag.as_simulation_core(bn, hie, root='country', random=['muA'], out=['foodstore', 'bmiA', 'bmiB'])
# sc = dag.SimulationCore(bn, bp, ng)

pc = sc.generate('Taiwan', {'sd': 1})
pc_taipei = pc.breed('Taipei', 'area')
pc_taipei.breed('A1', 'agA')
pc_taipei.breed('A2', 'agA')
pc_taipei.breed('B1', 'agB')
pc_taipei.breed('B2', 'agB')

pc.deep_print()
