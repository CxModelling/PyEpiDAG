import epidag as dag

__author__ = 'TimeWz667'


ca = dag.CompoundActor('B', [dag.ValueLoci('A', '0.01'),
                             dag.DistributionLoci('B', 'norm(A+10, 1)')])
print(ca.sample())


sa = dag.SingleActor('B', dag.DistributionLoci('B', 'norm(A+10, 1)'))
print(sa.sample({'A': 0.1}))


fsa = dag.FrozenSingleActor('B', dag.DistributionLoci('B', 'norm(A+10, 1)'), {'A': 0.1})
print(fsa.sample({'A': 0.1}))

