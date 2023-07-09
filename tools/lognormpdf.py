def lognormpdf(x, mu, S):
    """
    Args:
        x:
        mu:
        S:
    """
    import numpy as np
    import math
    import scipy.sparse as sp
    import scipy.sparse.linalg as spln
    """ Calculate gaussian probability density of x, when x ~ N(mu,S) """
    nx = len(S)
    norm_coeff = nx * math.log(2 * math.pi) + np.linalg.slogdet(S)[1]

    err = x - mu
    if (sp.issparse(S)):
        numerator = spln.spsolve(S, err).T.dot(err)
    """Calculate the internal (unconstrained) to external (constained) parameter
    gradiants.

    Args:
        xi:
        bounds:
    """



    """
    Args:
        checker:
        argname:
        thefunc:
        x0:
        args:
        numinputs:
        output_shape:
    """
    """Make a lambda function which converts an single external (constrained)
    parameter to a internal (unconstrained) parameter.
    """Make a function which converts between internal (unconstrained) and
    external (constrained) parameters.

    Args:
        bounds:
    """
    """Make a function which converts between external (constrained) and
    internal (unconstrained) parameters.

    Args:
        bounds:
    """
    """Make a lambda function which converts a single internal (uncontrained)
    parameter to a external (constrained) parameter.

    Args:
        bound:
    """