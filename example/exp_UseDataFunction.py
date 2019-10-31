import epidag as dag

__author__ = 'TimeWz667'


reg = {
    'Intercept': 0.5,
    'SE': 2.5,
    'Regressors': [
        {'Name': 'Age', 'Type': 'Continuous', 'Value': 5},
        {'Name': 'Male', 'Type': 'Boolean', 'Value': 5}
    ]
}


dag.add_data_func('lm', dag.data.LinearRegression(reg))


script = '''
PCore LM {
    Age ~ unif(0, 10)
    y ~ lm(Age, Male)
}
'''

bn = dag.bayes_net_from_script(script)

print(dag.sample(bn, {'Male': True}))
