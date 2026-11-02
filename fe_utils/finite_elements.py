import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle

np.seterr(invalid="ignore", divide="ignore")

def lagrange_points(cell, degree):
    # integer index tuples with x varying fastest: (0,0), (1,0), ..., (0,1), ...
    idx = [i[::-1] for i in np.ndindex(*(degree + 1,) * cell.dim)
           if sum(i) <= degree]

    def key(i):
        b = (degree - sum(i),) + tuple(i)  # barycentric coordinates
        nonzero = [k for k, x in enumerate(b) if x != 0]
        zero = [k for k, x in enumerate(b) if x == 0]
        if len(nonzero) == 1:   # vertex: ordered by which coordinate is nonzero
            return (1, nonzero[0])
        if zero:                # edge: grouped by the zero coordinate (opposite vertex)
            return (len(nonzero), zero[0])
        return (len(nonzero), 0)  # interior

    idx.sort(key=key)  # sort is stable, so the order within each group is kept
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
    return G.transpose(2, 0, 1)  # (points, monomials, gradient component)


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
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """

        return vandermonde_matrix(self.cell, self.degree, points) @ self.basis_coefs

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        nodes = lagrange_points(cell, degree)
        # Use lagrange_points to obtain the set of nodes. ...
        super(LagrangeElement, self).__init__(cell, degree, nodes)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.cell, self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell, degree)
        super(LagrangeElement, self).__init__(cell, degree, nodes)
