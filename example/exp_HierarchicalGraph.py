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

# Construct a Bayesian network from script
bj = dag.bn_script_to_json(scr)
bn = dag.BayesianNetwork(bj)

print(dag.sample(bn, {'sd': 1}))


hei = [[], ['b0r', 'ps', 'foodstore', 'pf'], ['bmiA', 'bmiB', 'ageA', 'ageB', 'sexA']]

print()
dag.form_hierarchy(bn, hei).print()


# define hierarchical levels
hie = {
    'root': ['area'],
    'area': ['b0r', 'ps', 'agA', 'agB', 'pf'],
    'agA': ['bmiA', 'ageA', 'sexA'],
    'agB': ['bmiB', 'ageB']
}

# locate all nodes
print()
root = dag.form_hierarchy(bn, hie)
root.print()

print()
dag.analyse_node_type(bn, root, True)

