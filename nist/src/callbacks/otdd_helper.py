# pylint: disable=W,C,R,E
import itertools
import sys

import geomloss
import ot
import torch
from tqdm.autonotebook import tqdm


def pwdist_exact(
    X1,
    Y1,
    X2=None,
    Y2=None,
    symmetric=False,
    loss="sinkhorn",
    cost_function="euclidean",
    p=2,
    debias=True,
    entreg=1e-1,
    device="cpu",
):
    """Efficient computation of pairwise label-to-label Wasserstein distances
    between multiple distributions, without using Gaussian assumption.
    Args:
        X1,X2 (tensor): n x d matrix with features
        Y1,Y2 (tensor): labels corresponding to samples
        symmetric (bool): whether X1/Y1 and X2/Y2 are to be treated as the same dataset
        cost_function (callable/string): the 'ground metric' between features to
            be used in optimal transport problem. If callable, should take follow
            the convection of the cost argument in geomloss.SamplesLoss
        p (int): power of the cost (i.e. order of p-Wasserstein distance). Ignored
            if cost_function is a callable.
        debias (bool): Only relevant for Sinkhorn. If true, uses debiased sinkhorn
            divergence.
    """

    if X2 is None:
        symmetric = True
        X2, Y2 = X1, Y1

    c1 = torch.unique(Y1)
    c2 = torch.unique(Y2)
    n1, n2 = len(c1), len(c2)

    # We account for the possibility that labels are shifted (c1[0]!=0), see below

    if symmetric:
        # If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        # If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    if cost_function == "euclidean":
        if p == 1:

            def cost_function(x, y):
                return geomloss.utils.distances(x, y)

        elif p == 2:

            def cost_function(x, y):
                return geomloss.utils.squared_distances(x, y)

        else:
            raise ValueError()

    if loss == "sinkhorn":
        distance = geomloss.SamplesLoss(
            loss=loss,
            p=p,
            cost=cost_function,
            debias=debias,
            blur=entreg ** (1 / p),
        )
    elif loss == "wasserstein":

        def distance(Xa, Xb):
            C = cost_function(Xa, Xb).cpu()
            # , verbose=True)
            return torch.tensor(ot.emd2(ot.unif(Xa.shape[0]), ot.unif(Xb.shape[0]), C))

    else:
        raise ValueError("Wrong loss")

    pbar = tqdm(pairs, leave=False)
    pbar.set_description("Computing label-to-label distances")
    D = torch.zeros((n1, n2), device=device, dtype=X1.dtype)
    for i, j in pbar:
        # try:
        D[i, j] = distance(
            X1[Y1 == c1[i]].to(device), X2[Y2 == c2[j]].to(device)
        ).item()
        # except:
        #     print(
        #         "This is awkward. Distance computation failed. Geomloss is hard to debug"
        #         "But here's a few things that might be happening: "
        #         " 1. Too many samples with this label, causing memory issues"
        #         " 2. Datatype errors, e.g., if the two datasets have different type"
        #     )
        #     sys.exit("Distance computation failed. Aborting.")
        if symmetric:
            D[j, i] = D[i, j]
    return D
