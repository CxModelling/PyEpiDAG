import unittest
import epidag as dag


class DAGTest(unittest.TestCase):
    def setUp(self):
        self.G = dag.DAG()
        self.G.add_edge('A', 'B')
        self.G.add_edge('B', 'C')
        self.G.add_edge('B', 'D')
        self.G.add_edge('C', 'D')

        self.Sub = dag.DAG()
        self.Sub.add_edge('E', 'D')

        self.G.nodes['A']['loci'] = [1, 2, 3]


    def test_relations(self):
        self.assertCountEqual(self.G.parents('D'), ['B', 'C'])
        self.assertCountEqual(self.G.ancestors('D'), ['A', 'B', 'C'])

    def test_copy(self):
        b = self.G.copy()
        b.nodes['A']['loci'][2] = 5
        self.assertSequenceEqual(self.G.nodes['A']['loci'], [1, 2, 5])

    def test_merge(self):
        mer = dag.merge_dag(self.G, self.Sub)
        self.assertCountEqual(mer.ancestors('D'), ['A', 'B', 'C', 'E'])

    def test_min_g(self):
        m = dag.minimal_dag(self.G, ['B', 'D'])
        self.assertIn('C', m)

    def test_min_needs(self):
        self.assertCountEqual(dag.minimal_requirements(self.G, 'B', []), ['A'])
        self.assertCountEqual(dag.minimal_requirements(self.G, 'D', []), ['A', 'B', 'C'])
        self.assertCountEqual(dag.minimal_requirements(self.G, 'D', ['B']), ['B', 'C'])



if __name__ == '__main__':
    unittest.main()
