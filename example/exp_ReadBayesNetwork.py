import epidag as dag


__author__ = 'TimeWz667'

scr = '''
    Pcore A{
        w = 1
        x1 = 1/x + exp(5)
        v ~ norm(z, sd)
        x = 0.2
        y ~ exp(x1)
        z ~ norm(w, y)
    }
    '''

bn = dag.bayes_net_from_script(scr)

print(bn)

print(dag.sample(bn, {'sd': 1}))
