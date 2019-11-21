import epidag as dag


__author__ = 'TimeWz667'


script = '''
PCore BetaBin {
    al = 1
    be = 1
    p ~ beta(al, be)
    x ~ binom(5, p)
}
'''


bn = dag.bayes_net_from_script(script)

ns = dag.NodeSet('root', as_fixed=['al', 'be'])
ns.new_child('ag', as_floating=['x'])

sc = dag.as_simulation_core(bn, ns)
sc.deep_print()

pc = sc.generate('a')
p = pc.breed('x', 'ag')

print(p.get_sampler('p'))
