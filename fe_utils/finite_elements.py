import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle

np.seterr(invalid="ignore", divide="ignore")


def lagrange_points(cell, degree):
    # integer index tuples with x varying fastest
    idx = [i[::-1] for i in np.ndindex(*(degree + 1,) * cell.dim)
           if sum(i) <= degree]

    def key(i):
        b = (degree - sum(i),) + tuple(i)  # barycentric coordinates
        nonzero = [k for k, x in enumerate(b) if x != 0]
        zero = [k for k, x in enumerate(b) if x == 0]
        if len(nonzero) == 1:   # vertex
            return (1, nonzero[0], 0)
        if zero:                # edge: grouped by the zero coordinate
            if len(nonzero) == 2 and zero[0] == 0:
                return (2, 0, -i[0])   # edge 0 runs from (1,0) to (0,1)
            return (len(nonzero), zero[0], 0)
        return (len(nonzero), 0, 0)    # interior

    idx.sort(key=key)
    return np.array(idx) / degree


def vandermonde_matrix(cell, degree, points, grad=False):
    points = np.asarray(points)
    if cell.dim == 1:
        exps = [(d,) for d in range(degree + 1)]
    else:
        exps = [(d - j, j) for d in range(degree + 1) for j in range(d + 1)]

    def monomial(e):
        return np.prod(points ** np.array(e), axis=1)

    if not grad:
        return np.stack([monomial(e) for e in exps], axis=1)

    def dmonomial(e, k):
        if e[k] == 0:
            return np.zeros(len(points))
        e2 = list(e)
        e2[k] -= 1
        return e[k] * monomial(e2)

    G = np.array([[dmonomial(e, k) for k in range(cell.dim)] for e in exps])
    return G.transpose(2, 0, 1)


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """(keep the docstring as it was)"""

        self.cell = cell
        self.degree = degree
        self.nodes = nodes
        self.entity_nodes = entity_nodes

        if entity_nodes:
            self.nodes_per_entity = np.array(
                [len(entity_nodes[d][0]) for d in range(cell.dim + 1)]
            )

        self.basis_coefs = np.linalg.inv(vandermonde_matrix(cell, degree, nodes))

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """(keep the docstring as it was)"""

        V = vandermonde_matrix(self.cell, self.degree, points, grad)
        if grad:
            return np.einsum("ijk,jl->ilk", V, self.basis_coefs)
        return V @ self.basis_coefs

    def interpolate(self, fn):
        """(keep the docstring as it was)"""

        return np.array([fn(x) for x in self.nodes])

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.cell, self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """(keep the docstring as it was)"""

        nodes = lagrange_points(cell, degree)

        def support(p):
            b = (1 - p.sum(),) + tuple(p)  # barycentric coordinates
            return {k for k, x in enumerate(b) if x > 1e-12}

        entity_nodes = {
            d: {i: [n for n, p in enumerate(nodes)
                    if support(p) == set(cell.topology[d][i])]
                for i in range(len(cell.topology[d]))}
            for d in range(cell.dim + 1)
        }
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes)