import epidag as dag
import numpy as np

__author__ = 'TimeWz667'


script = '''
PCore Exp2 {
   mu_x = 0
   mu_y = 10
   sd ~ exp(1)
   x ~ norm(mu_x, sd)
   y ~ norm(mu_y, sd)
}
'''

bn = dag.bayes_net_from_script(script)
sc = dag.as_simulation_core(bn)
sc.deep_print()
sg = sc.generate('exp2')

print(sg)
y = sg.get_sampler('y')
print(np.mean([y() for _ in range(1000)]))

sg.impulse({'mu_y': 100})
print(sg)
print(np.mean([y() for _ in range(1000)]))
