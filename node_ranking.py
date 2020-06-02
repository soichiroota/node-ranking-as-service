import numpy as np
from scipy import sparse
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power
from sknetwork.ranking import (
    PageRank, Diffusion, HITS, Closeness, Harmonic
)
from sknetwork.utils import edgelist2adjacency


class NodeScorer:
    def __init__(
        self,
        library=None,
        algo=None,
        damping_factor=0.85,
        solver='naive',
        n_iter=10,
        tol=0,
        undirected=False,
        method='exact'
    ):
        if library == 'scikit-network':
            self.scorer = ScikitNetworkScorer(
                algo=algo,
                damping_factor=damping_factor,
                solver=solver,
                n_iter=n_iter,
                tol=tol,
                undirected=undirected,
                method=method
            )
        else:
            self.scorer = FastPageRankScorer(
                solver=solver,
                damping_factor=damping_factor,
                tol=tol
            )

    def score(self, data):
        return self.scorer.score(data)


class FastPageRankScorer:
    def __init__(self, damping_factor=0.85, solver='power', tol=0):
        self.damping_factor = damping_factor
        self.solver = solver
        self.tol = tol

    def score(self, data):
        node_dict = {
            node['name']: i for i, node in enumerate(data['node'])
        }
        node_count = len(node_dict)
        edges = np.array(
            [(
                node_dict[edge['node'][0]],
                node_dict[edge['node'][1]]
            ) for edge in data['edge']]
        )
        weights = np.array([edge['weight'] for edge in data['edge']])
        G = sparse.csr_matrix(
            (weights, (edges[:,0], edges[:,1])),
            shape=(node_count, node_count)
        )
        if data['node'][0].get('weight'):
            personalize = np.array(
                [node['weight'] for node in data['node']]
            )
        else:
            personalize = None
        if self.solver == 'power':
            pr = pagerank_power(
                G,
                p=self.damping_factor,
                personalize=personalize,
                tol=self.tol
            )
        else:
            pr = pagerank(
                G,
                p=self.damping_factor,
                personalize=personalize
            )
        return {k: pr[v] for k, v in node_dict.items()}


class ScikitNetworkScorer:
    def __init__(
        self,
        algo=None,
        damping_factor=0.85,
        solver='naive',
        n_iter=10,
        tol=0,
        undirected=False,
        method='exact'
    ):
        if algo == 'diffusion':
            self.scorer = Diffusion(n_iter=n_iter)
        elif algo == 'closeness':
            self.scorer = Closeness(method=method, tol=tol)
        elif algo == 'harmonic':
            self.scorer = Harmonic()
        else:
            self.scorer = PageRank(
                damping_factor=damping_factor,
                solver=solver,
                n_iter=n_iter,
                tol=tol
            )
        self.undirected = undirected

    def score(self, data):
        node_dict = {
            node['name']: i for i, node in enumerate(data['node'])
        }
        edges = np.array(
            [(
                node_dict[edge['node'][0]],
                node_dict[edge['node'][1]],
                edge['weight']
            ) for edge in data['edge']]
        )
        adjacency = edgelist2adjacency(
            edges,
            undirected=self.undirected
        )
        if data['node'][0].get('weight'):
            seeds = np.array(
                [node['weight'] for node in data['node']]
            )
            scores = self.scorer.fit_transform(adjacency, seeds)
        else:
            scores = self.scorer.fit_transform(adjacency)
        return {k: scores[v] for k, v in node_dict.items()}
