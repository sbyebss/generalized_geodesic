import sys
import logging
import pdb
import gc
import itertools
import numpy as np
import math
import torch
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
import geomloss
import ot
from time import time

from .sqrtm import sqrtm, sqrtm_newton_schulz
from .utils import process_device_arg, infer_torch_object_type, load_full_dataset,extract_data_targets
from .utils import safe_next_iter, seed_worker
from .datasets import  SubsetFromLabels, InfiniteDataLoader

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

MAXBATCH=4000
DEBUG=False
PROFILING=False

def bures_distance(Σ1, Σ2, sqrtΣ1, commute=False, squared=True):
    """ Bures distance between PDF matrices. Simple, non-batch version.
        Potentially deprecated.
    """
    if not commute:
        sqrtΣ1 = sqrtΣ1 if sqrtΣ1 is not None else sqrtm(Σ1)
        bures = torch.trace(
            Σ1 + Σ2 - 2 * sqrtm(torch.mm(torch.mm(sqrtΣ1, Σ2), sqrtΣ1)))
    else:
        #bures = torch.norm(sqrtm(Σ1) - sqrtm(Σ2), ord='fro')**2
        bures = ((sqrtm(Σ1) - sqrtm(Σ2))**2).sum()
    if not squared:
        bures = torch.sqrt(bures)
    return torch.relu(bures)  # i.e., max(bures,0)


def bbures_distance(Σ1, Σ2, sqrtΣ1=None, inv_sqrtΣ1=None,
                    diagonal_cov=False, commute=False, squared=True, sqrt_method='spectral',
                    sqrt_niters=20):
    """ Bures distance between PDF. Batched version. """
    if sqrtΣ1 is None and not diagonal_cov:
        sqrtΣ1 = sqrtm(Σ1) if sqrt_method == 'spectral' else sqrtm_newton_schulz(Σ1, sqrt_niters)  # , return_inverse=True)

    if diagonal_cov:
        bures = ((torch.sqrt(Σ1) - torch.sqrt(Σ2))**2).sum(-1)
    elif commute:
        sqrtΣ2 = sqrtm(Σ2) if sqrt_method == 'spectral' else sqrtm_newton_schulz(Σ2, sqrt_niters)
        #bures = torch.norm(sqrtm(Σ1) - sqrtm(Σ2), ord='fro')**2
        bures = ((sqrtm(Σ1) - sqrtm(Σ2))**2).sum((-2, -1))
    else:
        if sqrt_method == 'spectral':
            cross = sqrtm(torch.matmul(torch.matmul(sqrtΣ1, Σ2), sqrtΣ1))
        else:
            cross = sqrtm_newton_schulz(torch.matmul(torch.matmul(
                sqrtΣ1, Σ2), sqrtΣ1), sqrt_niters)
        ## pytorch doesn't have batched trace yet!
        bures = (Σ1 + Σ2 - 2 * cross).diagonal(dim1=-2, dim2=-1).sum(-1)
    if not squared:
        bures = torch.sqrt(bures)
    # if sqrtΣ1.requires_grad:
    #     register_gradient_hook(cross, 'cross')
    #     register_gradient_hook(sqrtΣ1, 'sqrtΣ1')
    #     register_gradient_hook(Σ1, 'Σ1')
    return torch.relu(bures)


def wasserstein_gauss_distance(μ_1, μ_2, Σ1, Σ2, sqrtΣ1=None, cost_function='euclidean',
                               squared=False,**kwargs):
                               # sqrt_method='spectral', sqrt_niters=20,
                               # commute=False, diagonal_cov=False):
    """
    Returns 2-Wasserstein Distance between Gaussians:

         W(α, β)^2 = || μ_α - μ_β ||^2 + Bures(Σ_α, Σ_β)^2


    Arguments:
        μ_1 (tensor): mean of first Gaussian
        kwargs (dict): additional arguments for bbures_distance.

    Returns:
        d (tensor): the Wasserstein distance

    """
    if cost_function == 'euclidean':
        mean_diff = ((μ_1 - μ_2)**2).sum(axis=-1)  # I think this is faster than torch.norm(μ_1-μ_2)**2
    else:
        mean_diff = cost_function(μ_1,μ_2)
        pdb.set_trace(header='TODO: what happens to bures distance for embedded cost function?')

    cova_diff = bbures_distance(Σ1, Σ2, sqrtΣ1=sqrtΣ1, squared=True, **kwargs)
    d = torch.relu(mean_diff + cova_diff)
    if not squared:
        d = torch.sqrt(d)
    return d


def process_dist_args(*args):
    """ Helper function to process and infer types of positional args passed to
    pwdist """
    DS1 = DS2 = X1 = X2 = Y1 = Y2 = None
    assert len(args) >= 1
    type_1 = infer_torch_object_type(args[0])
    if len(args) == 4: #X1,Y1,X2,Y2 passed, should be all tensors
        type_2 = infer_torch_object_type(args[2])
        assert type_1 == type_2 # mixed data types not accepted for now
        X1, Y1, X2, Y2 = args
        symmetric = False
    elif len(args) == 2: # either two dataloaders, or X1,Y1 for symmetric, or dict if passed as paths
        if type_1 == 'tensor':
            X1, Y1 = args
            X2, Y2 = X1, Y1
            symmetric = True
        elif type_1 == 'dict':
            type_2 = infer_torch_object_type(args[1])
            assert type_1 == type_2 # mixed data types not accepted for now
            symmetric = False
        else:
            type_2 = infer_torch_object_type(args[1])
            assert type_1 == type_2 # mixed data types not accepted for now
            DS1, DS2 = args
            symmetric = False
    elif len(args) == 1:
        assert type_1 in ['dataloader', 'dataset', 'dict']
        DS1 = args[0]
        DS2 = DS1
        symmetric = True
    return X1,Y1,X2,Y2,DS1,DS2,type_1,symmetric


def pwdist_gauss(M1, S1, M2, S2, symmetric=False, return_dmeans=False, nworkers=1,
                 commute=False):
    """ POTENTIALLY DEPRECATED.
        Computes Wasserstein Distance between collections of Gaussians,
        represented in terms of their means (M1,M2) and Covariances (S1,S2).

        Arguments:
            parallel (bool): Whether to use multiprocessing via joblib


     """
    n1, n2 = len(M1), len(M2)  # Number of clusters in each side
    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    D = torch.zeros((n1, n2)).to(device)

    if nworkers > 1:
        # CONSERVATIVE METHOD: return (i,j) to make sure we don't mix them up. Requires defining wrapper fun.
        # def _wrapfun(i,j,μ1, μ2, Σ1, Σ2):
        #     return ((i,j),wasserstein_gauss_distance(μ1, μ2, Σ1, Σ2, squared=True))
        # results = Parallel(n_jobs=nworkers, verbose=1, backend="threading")(
        #             delayed(_wrapfun)(i,j,M1[i],M2[j],S1[i],S2[j]) for i,j in pairs)
        # for (i,j),d in results:
        #     D[i,j] = d
        #     if symmetric: D[j,i] = D[i,j]
        # CLEANER/RISKIER METHOD: If Parallel returns arguments in order (??), can do this instead: CHECK
        results = Parallel(n_jobs=nworkers, verbose=1, backend="threading")(
            delayed(wasserstein_gauss_distance)(M1[i], M2[j], S1[i], S2[j], squared=True) for i, j in pairs)
        for (i, j), d in zip(pairs, results):
            D[i, j] = d
            if symmetric:
                D[j, i] = D[i, j]
    else:
        for i, j in tqdm(pairs, leave=False):
            D[i, j] = wasserstein_gauss_distance(
                M1[i], M2[j], S1[i], S2[j], squared=True, commute=commute)
            if symmetric:
                D[j, i] = D[i, j]

    if return_dmeans:
        D_means = torch.cdist(M1, M2)  # For viz purposes only
        return D, D_means
    else:
        return D


def efficient_pwdist_gauss(M1, S1, M2=None, S2=None, sqrtS1=None, sqrtS2=None,
                        symmetric=False, diagonal_cov=False, commute=False,
                        sqrt_method='spectral',sqrt_niters=20,sqrt_pref=0,
                        device='cpu',nworkers=1,
                        cost_function='euclidean',
                        return_dmeans=False, return_sqrts=False):
    """ [Formerly known as efficient_pwassdist] Efficient computation of pairwise
    label-to-label Wasserstein distances between various distributions. Saves
    computation by precomputing and storing covariance square roots."""
    if M2 is None:
        symmetric = True
        M2, S2 = M1, S1

    n1, n2 = len(M1), len(M2)  # Number of clusters in each side
    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    D = torch.zeros((n1, n2), device = device, dtype=M1.dtype)

    sqrtS = []
    ## Note that we need inverses of only one of two datasets.
    ## If sqrtS of S1 provided, use those. If S2 provided, flip roles of covs in Bures
    both_sqrt = (sqrtS1 is not None) and (sqrtS2 is not None)
    if (both_sqrt and sqrt_pref==0) or (sqrtS1 is not None):
        ## Either both were provided and S1 (idx=0) is prefered, or only S1 provided
        flip = False
        sqrtS = sqrtS1
    elif sqrtS2 is not None:
        ## S1 wasn't provided
        if sqrt_pref == 0: logger.warning('sqrt_pref=0 but S1 not provided!')
        flip = True
        sqrtS = sqrtS2  # S2 playes role of S1
    elif len(S1) <= len(S2):  # No precomputed squareroots provided. Compute, but choose smaller of the two!
        flip = False
        S = S1
    else:
        flip = True
        S = S2  # S2 playes role of S1

    if not sqrtS:
        logger.info('Precomputing covariance matrix square roots...')
        for i, Σ in tqdm(enumerate(S), leave=False):
            if diagonal_cov:
                assert Σ.ndim == 1
                sqrtS.append(torch.sqrt(Σ)) # This is actually not needed.
            else:
                sqrtS.append(sqrtm(Σ) if sqrt_method ==
                         'spectral' else sqrtm_newton_schulz(Σ, sqrt_niters))

    logger.info('Computing gaussian-to-gaussian wasserstein distances...')
    pbar = tqdm(pairs, leave=False)
    pbar.set_description('Computing label-to-label distances')
    for i, j in pbar:
        if not flip:
            D[i, j] = wasserstein_gauss_distance(M1[i], M2[j], S1[i], S2[j], sqrtS[i],
                                                 diagonal_cov=diagonal_cov,
                                                 commute=commute, squared=True,
                                                 cost_function=cost_function,
                                                 sqrt_method=sqrt_method,
                                                 sqrt_niters=sqrt_niters)
        else:
            D[i, j] = wasserstein_gauss_distance(M2[j], M1[i], S2[j], S1[i], sqrtS[j],
                                                 diagonal_cov=diagonal_cov,
                                                 commute=commute, squared=True,
                                                 cost_function=cost_function,
                                                 sqrt_method=sqrt_method,
                                                 sqrt_niters=sqrt_niters)
        if symmetric:
            D[j, i] = D[i, j]

    if return_dmeans:
        D_means = torch.cdist(M1, M2)  # For viz purposes only
        if return_sqrts:
            return D, D_means, sqrtS
        else:
            return D, D_means
    elif return_sqrts:
        return D, sqrtS
    else:
        return D

def pwdist_means_only(M1, M2=None, symmetric=False, device=None):
    if M2 is None or symmetric:
        symmetric = True
        M2 = M1
    D = torch.cdist(M1, M2)
    if device:
        D = D.to(device)
    return D

def pwdist_upperbound(M1, S1, M2=None, S2=None,symmetric=False, means_only=False,
                          diagonal_cov=False, commute=False, device=None,
                          return_dmeans=False):
    """ Computes upper bound of the Wasserstein distance between distributions
    with given mean and covariance.
    """

    if M2 is None:
        symmetric = True
        M2, S2 = M1, S1

    n1, n2 = len(M1), len(M2)  # Number of clusters in each side
    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    D = torch.zeros((n1, n2), device = device, dtype=M1.dtype)

    logger.info('Computing gaussian-to-gaussian wasserstein distances...')
    pbar = tqdm(pairs, leave=False)
    pbar.set_description('Computing label-to-label distances')

    if means_only or return_dmeans:
        D_means = torch.cdist(M1, M2)

    if not means_only:
        for i, j in pbar:
            if means_only:
                D[i,j] = ((M1[i]-  M2[j])**2).sum(axis=-1)
            else:
                D[i,j] = ((M1[i]-  M2[j])**2).sum(axis=-1) + (S1[i] + S2[j]).diagonal(dim1=-2, dim2=-1).sum(-1)
            if symmetric:
                D[j, i] = D[i, j]
    else:
        D = D_means

    if return_dmeans:
        D_means = torch.cdist(M1, M2)  # For viz purposes only
        return D, D_means
    else:
        return D

def pwdist_exact(*args, symmetric=False, loss='sinkhorn', p=2, debias=True,
                  entreg=1e-1, cost_function='euclidean', batchified=False,
                  maxbatch=1024, minbatch=128, nbatches='dynamic', nworkers=None,
                  caching='both', persist_dataloaders=True, max_dataloaders = 100,
                  persist_factor = 1.0, device='cpu'):
    """ Efficient computation of pairwise label-to-label Wasserstein distances
    between multiple distributions, without using Gaussian assumption.

    Args:
        args: inputs
        symmetric (bool): whether X1/Y1 and X2/Y2 are to be treated as the same dataset
        cost_function (callable/string): the 'ground metric' between features to
            be used in optimal transport problem. If callable, should take follow
            the convection of the cost argument in geomloss.SamplesLoss
        p (int): power of the cost (i.e. order of p-Wasserstein distance). Ignored
            if cost_function is a callable.
        debias (bool): Only relevant for Sinkhorn. If true, uses debiased sinkhorn
            divergence.
        caching (str): Whether to cache tensors after loading. The options are None 
            (not caching), 'left' (cache only the left side), 'both' (cache both sides)


    The DatasetDistance eager_data_loading and batchified can be used to choose
    among the following behaviors:
        1. If tensors are passed (a consequence of EDL), computation will rely
        on slicing, batchified will be ignored.
        2. If datasets are passed (non EDL) and batchified=False, will create per-
        class datasets and them load them entirely to memory, two at a time, to
        compute exact SamplesLoss distance between them.
        3. If datasets are passed (non EDL) and batchified=True, will create per-
        class datasets but will not load them to memory, instead relying on
        BatchifiedSamplesLoss to compute approximate (batchified) distance.

    """
    device = process_device_arg(device)

    X1,Y1,X2,Y2,DS1,DS2,input_type,symmetric = process_dist_args(*args)

    assert not (input_type in ['tensor','dataloader'] and batchified),  \
    'Batchified distance on Tensor/DataLoader not supported yet'

    if input_type  == 'tensor':
        c1 = torch.unique(Y1)
        c2 = torch.unique(Y2)
        n1, n2 = len(c1), len(c2)
        if DEBUG: print(symmetric, p, debias, entreg, X1.shape, X2.shape)
    elif input_type == 'dataset':
        # TODO: have the option to pass set of classes to avoid this
        _, c1, _ = extract_data_targets(DS1)
        _, c2, _ = extract_data_targets(DS2)
        n1, n2 = len(c1), len(c2)
    elif input_type == 'dataloader':
        _, c1, _ = extract_data_targets(DS1)
        _, c2, _ = extract_data_targets(DS2)
        n1, n2 = len(c1), len(c2)
    elif input_type == 'dict':
        n1, n2 = len(DS1), len(DS2)

    # If either dataset has fewer samples per class than maxbatch, no need to batchify that side
    batchified_a = batchified and (len(DS1)/n1 > maxbatch)
    batchified_b = batchified and (len(DS2)/n2 > maxbatch)

    if batchified and not (batchified_a or batchified_b):
        logger.warning('Batchified distance requested but not needed, reverting to non-batchified')
        batchified = False

    ## We account for the possibility that labels are shifted (c1[0]!=0), see below
    if symmetric:
        ## If tasks are symmetric (same data on both sides) only need combinations
        pairs = list(itertools.combinations(range(n1), 2))
    else:
        ## If tasks are assymetric, need n1 x n2 comparisons
        pairs = list(itertools.product(range(n1), range(n2)))

    # FIXME: geomloss (and the other cost_routines fun) devide by p. Don't want that here
    # to be able to easily compare across inner_ot methods
    if cost_function == 'euclidean':
        if p == 1:
            cost_function = lambda x, y: geomloss.utils.distances(x, y)
        elif p == 2:
            cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
        else:
            raise ValueError()
        embedder = None
    else:
        assert cost_function.src_emb == cost_function.tgt_emb, 'BatchifiedSamplesLoss assumes same src/tgt embedder'
        embedder = cost_function.src_emb

    if loss == 'sinkhorn':
        if batchified:
            #pdb.set_trace(header='This is not implemnented for batchified yet')
            distance = BatchifiedSamplesLoss(
                # BatchifiedSamplesLoss-specific Args
                maxbatch=maxbatch,
                minbatch=minbatch,
                shuffle=False,
                on_joint=False,
                nworkers = nworkers,
                embedder = embedder,
                # Regular SamplesLoss Args
                loss=loss, p=p,
                debias=debias,
                blur=entreg**(1 / p),
                backend='tensorized', # For batchified, can afford to do tensorized
                device=device
            )
        else:
            ### FIXME: how to deal with non-euclidean cost_function here?
            ### Option 1: pass cost here (uncomment)
            ### Option 2: embed when using _load_dataloader below
            #assert cost_function == 'euclidean', 'non-batchified pwdist with custom feature cost not implemented yet'
            distance = geomloss.SamplesLoss(
                loss=loss, p=p,
                #cost=None if cost_function == 'euclidean' else cost_function, # no need to specify if cost is vanilla, plus will break if backend!=tensorized
                debias=debias,
                blur=entreg**(1 / p),
                backend='tensorized',
                # FIXME: we don't tensorized need here, since only on features, but it's failing witnh online for me
            )
    elif loss == 'wasserstein':
        def distance(Xa, Xb):
            C = cost_function(Xa, Xb).cpu()
            #C2 = ot.dist(Xa,Xb, metric='sqeuclidean')
            return torch.tensor(ot.emd2(ot.unif(Xa.shape[0]), ot.unif(Xb.shape[0]), C))#, verbose=True)
    else:
        raise ValueError('Unrecognized loss type')

    # TODO: If computational graph is not broken, we could even do this conversion
    # insinde the for loop, only the classes being processed need to be in GPU

    logger.info('Computing label-to-label (exact) wasserstein distances...')
    pbar = tqdm(pairs, leave=False)
    pbar.set_description('Computing label-to-label distances')
    D = torch.zeros((n1,n2), device=device, dtype=X1.dtype if input_type=='tensor' else None)


    def _load_dataloader(loader, device='cpu', flatten=True, embedding=None):
        ''' Minimalistic function to load entire dataset to tensor '''
        X,Y,nb = [],[],0
        if embedding is not None:
            embedding = embedding.to(device)
        for x,y in loader: # Can we do this with list comprehension?
            with torch.no_grad():
                if embedding is None:
                    X.append((x.view(x.shape[0], -1) if flatten else x).to('cpu'))
                else:
                    X.append(embedding(x.to(device)).view(x.shape[0], -1).to('cpu'))
            Y.append(y.to('cpu'))
            nb += 1
            if nb == len(loader): break # To account for InfiniteDataLoaders #TODO: I think this might not be necessary. Check.
        return torch.cat(X), torch.cat(Y)

    ## We will have three types of data loaders:
    ## - load_disp: for entire loading into memory disposable dataloaders
    ## - load_pers: for entire loading into memory persistent dataloaders
    ## - lazy_pers: for lazy batch-by-batch loading of persistent dataloaders
    load_disp_dlargs = {'batch_size': 512, 'num_workers': nworkers, 'pin_memory': True,
                       'drop_last': False}#, 'persistent_workers': False}
    load_pers_dlargs = load_disp_dlargs.copy()
    load_pers_dlargs.update({'persistent_workers': nworkers>100})
    if nworkers > 0:
        load_pers_dlargs.update({'prefetch_factor': 1})
    lazy_pers_dlargs  = {'batch_size': maxbatch, 'num_workers': nworkers, 'pin_memory': False, #'prefetch_factor': 1000,
                        'persistent_workers': nworkers>100} #'drop_last': False,  will be decided later, case-by-case

    _loaders = [{},{}]
    _tensors = [{},{}]
    prev_i = None

    # TODO: Preintiailizing dataloaders here prevents us from doing dynamic batch sizing
    # can we circumvent this by caching first dataset as tensor
    for i, j in pbar:
        t0 = time()
        try:
            #pdb.set_trace()
            if input_type == 'dict':
                if embedder is not None:
                    logger.warning('Cannot guarantee that data has already been embedded')
                    breakpoint()
                ''' Per-class data has been stored as tensors '''
                X1_i = torch.load(DS1[i], map_location=device)
                X2_j = torch.load(DS2[j], map_location=device)
                dist_inputs = (X1_i.view(X1_i.shape[0], -1), X2_j.view(X2_j.shape[0], -1))
            elif input_type == 'tensor':
                ''' Data is already in memory, so might as well run full-batch OT '''
                X1_i = X1[Y1==c1[i]] # Using Y1==c1[i] etc accounts for shifted labels
                X2_j = X2[Y2==c2[j]]
                dist_inputs = (X1_i.to(device), X2_j.to(device))
            elif input_type == 'dataloader':
                X1_i,Y1_i = _load_dataloader(DataLoader(SubsetFromLabels(DS1.dataset, [i]),
                                             **load_disp_dlargs), device=device,
                                             embedding = embedder, flatten=True)
                X2_j,Y2_j = _load_dataloader(DataLoader(SubsetFromLabels(DS2.dataset, [j]),
                                             **load_disp_dlargs), device=device,
                                             embedding = embedder, flatten=True)
                dist_inputs = (X1_i.to(device), X2_j.to(device))
            elif input_type == 'dataset' and not batchified:
                ''' Data not loaded yet, but don't want to batchify. To avoid
                    slow first-batch loading, we always keep one subdataset loaded
                    onto memory, and keep the dataloaders of the other dataset alive.

                    TODO: might want to decide which dataset is loaded dynamically,
                    based on number of samples per class.
                '''
                # TODO: Instead of caching, could just decouple for loop into nested loops
                if not caching or prev_i != i:
                    logger.info("Loading tensor to memory (left side) in non-batchified label-to-label dist computation!")
                    X1_i,Y1_i = _load_dataloader(DataLoader(SubsetFromLabels(DS1, [i]),
                                                 **load_disp_dlargs), device=device,
                                                 embedding = embedder, flatten=True)
                else:
                    logger.info("Reusing cached tensor (left side) in non-batchified label-to-label dist computation!")
                prev_i = i

                if caching == 'both':
                    if j in _tensors[1]:# we already loaded this tensor before into cpu memory
                        logger.info("Reusing cached tensor (right side) in non-batchified label-to-label dist computation!")
                        X2_j,Y2_j = _tensors[1][j]
                    else:
                        logger.info("Loading tensor to memory (right side) in non-batchified label-to-label dist computation!")
                        X2_j,Y2_j = _load_dataloader(DataLoader(SubsetFromLabels(DS2, [j]),
                                                    **load_disp_dlargs), device=device,
                                                    embedding = embedder, flatten=True)        
                        _tensors[1][j] = (X2_j,Y2_j)                
                elif j in _loaders[1]: # we already init'd this dataloder, retrieve
                    logger.info("Reusing persisent dl in non-batchified label-to-label dist computation!")
                    X2_j, Y2_j = _load_dataloader(_loaders[1][j], device=device, embedding = embedder,flatten=True)
                elif persist_dataloaders and (len(_loaders[1]) < max_dataloaders) and (persist_factor > np.random.rand()):
                    logger.info("Creating persisent dl in non-batchified label-to-label dist computation!")
                    _loaders[1][j] = DataLoader(SubsetFromLabels(DS2, [j]), **load_pers_dlargs)
                    X2_j, Y2_j = _load_dataloader(_loaders[1][j], device=device, embedding = embedder, flatten=True)
                else:
                    logger.info("Creating disposable dl in non-batchified label-to-label dist computation!")
                    X2_j, Y2_j= _load_dataloader(DataLoader(SubsetFromLabels(DS2, [j]),
                                                 **load_disp_dlargs), device=device, embedding = embedder, flatten=True)

                # if persist_dataloaders and (not j in _loaders[1]):
                #     _loaders[1][j] = DataLoader(SubsetFromLabels(DS2, [j]), **load_pers_dlargs)
                #     X2_j, Y2_j = _load_dataloader(_loaders[1][j], device=device, flatten=True)
                # else:
                assert len(torch.unique(Y1_i))==1 and len(torch.unique(Y2_j))==1, "Error: Mixed labels detected in batch"
                dist_inputs = (X1_i.to(device), X2_j.to(device)) # .to should not be necessary here if we pass to _load
            elif input_type == 'dataset' and batchified:
                ''' Data not loaded yet, we will load batch by batch in samplesloss.
                    We avoid repeated initialization of both sets of subdatasets
                    dataloaders for the same reasons as above.
                '''
                logger.debug("input_type == 'dataset' and batchified:")
                if batchified_a and not i in _loaders[0]:
                    #_loaders[0][i] = InfiniteDataLoader(SubsetFromLabels(DS1, [i]), **lazy_pers_dlargs) # Hangs!
                    _loaders[0][i] = DataLoader(dset:=SubsetFromLabels(DS1, [i]), **lazy_pers_dlargs, drop_last= len(dset) % maxbatch < minbatch)
                    logger.info("Creating new dataloader for dataset A label %d in batchified label-to-label dist computation!" % i)
                else:
                    if not caching or prev_i != i:
                        X1_i,Y1_i = _load_dataloader(DataLoader(SubsetFromLabels(DS1, [i]),
                                                    **load_disp_dlargs), device=device,
                                                    embedding = embedder, flatten=True)
                        X1_i = X1_i.to(device)
                        Y1_i = Y1_i.to(device)
                        logger.info("Loading tensors to memory in batchified label-to-label dist computation!")                        
                    else:
                        logger.info("Reusing cache in batchified label-to-label dist computation!")
                    prev_i = i

                if not j in _loaders[1]:
                    #_loaders[1][j] = InfiniteDataLoader(SubsetFromLabels(DS2, [j]), **lazy_pers_dlargs) # Hangs!
                    _loaders[1][j] = DataLoader(dset:=SubsetFromLabels(DS2, [j]), **lazy_pers_dlargs,  drop_last= bool(len(dset) % maxbatch < minbatch))

                dist_inputs = (_loaders[0][i] if batchified_a else (X1_i,Y1_i), _loaders[1][j])

            t1 = time()
            logger.info(f'Time to prepare geomloss inputs: {t1-t0:8.4e}s')

            D[i, j] = distance(*dist_inputs).item()
            t2 = time()
            logger.info(f'Time to compute distance: {t2-t1:8.4e}s')

            if not batchified:
                n,m = len(dist_inputs[0]), len(dist_inputs[1]) # Works for tensor *or* dataset
            else:
                n,m = distance.seen_samples['a'], distance.seen_samples['b']

            if torch.is_tensor(dist_inputs[0]) and torch.is_tensor(dist_inputs[1]):
                d1, d2 = dist_inputs[0].shape[1], dist_inputs[1].shape[1]
                logger.info(f'Dimensions of samples used in inner-OT problem: {d1,d2}')
            logger.info(f'Effective # of samples used in inner-OT problem: {n,m}')
            # del X2_j
            # del dist_inputs
            # _loaders[1].pop(j,None)
        except Exception as e:
            logger.error(f'Label-to-label distance pair ({i},{j}) failed with error: '+str(e))
            breakpoint()
            sys.exit('Distance computation failed. Aborting.')
        if symmetric:
            D[j, i] = D[i, j]

        gc.collect()
        logger.info(f'Current persisted dataloaders: {len(_loaders[1])}')
    #pdb.set_trace()
    return D


class BatchifiedSamplesLoss(torch.nn.Module):
    """ A wrapper around geomloss' SamplesLoss that computes the loss by batches.

        Arguments:
            maxbatch (int): maximum allowed batch size
            minbatch (int): minimum allowed batch size
            nbatches (str,int): Controls how large and how many batches are used.
                If 'dynamic', chooses number and size of batches dynamically so
                that approximately all samples from both datasets are consumed
                in the same number of batches by choosing different batch sizes
                for the two datasets. Otherwise, batch size will be same for both
                datasets (=maxbatch), and the number of batches is determined as
                follows. If 'min', will load as many batches as the smaller
                of the two datasets has, ignoring the rest from the larger one.
                If 'max' (default), will load all batches from larger of the two, re-
                initializing the smaller one as needed. If int, will load those
                many batches, reinitializing both dataloaders as needed.
            on_joint (bool): whether the loss is to be computed on join (x,y) samples
            embedder (torch Module): if given, will be used to embed batch featues
                before computing distance
            shuffle (bool): whether to shuffle the dataloaders
            nworkers (int): number of workers for the dataloaders
            device (str): device where computation will be carried
            Kwargs (dict): additional arguments to be passed to batch-wise SamplesLoss
                note, in particular, that custom cost functions are passed.

        Notation:
            N_[a|b]: total number of samples in dataset a / dataset b
            B_[a|b]: samples per batch in dataset a / dataset b
            NB_[a|b]: number of batches in dataset a / dataset b

        Notes:
        -  Advantages of passing dla, dlb as datasets:
             1. Freedom to choose batch size for each to ensure that
                K = N_x/B_x = N_y/B_y, ie *all* samples from both are consumed
                in exactly K batches

        TODOS:
            - Consider alternatives to my dataset recyling strategy:
                1. Using itertools.cycle on the smaller dataset (~sampling with replacement from smaller)
                2. Using itertools.cycle on both (~sampling with replacement from both),
                   and have a fixed number of iters defined in advance.
              I read somewhere that pytorch DataLoaders can have memory leaks when
              used with cycle. Check.
    """
    def __init__(self, *args, maxbatch=1024, minbatch=128, nbatches='max',
                 on_joint=False, shift_ys = (None,None), embedder=None, shuffle=True, nworkers=32,
                 multiprocessing_context='fork', pin_memory=True, infinite_loaders=True,
                 persistent_workers=True,
                 device='cpu', seed=None, **kwargs):
        super(BatchifiedSamplesLoss, self).__init__()
        self.batchloss = geomloss.SamplesLoss(*args, **kwargs)
        self.device    = device
        self.maxbatch  = maxbatch # Maybe: determine largest possible batch on the fly?
        self.minbatch  = minbatch
        self.nbatches  = nbatches
        self.on_joint  = on_joint
        self.shift_ys  = shift_ys # Needed to make debiased batch_cost work
        ### TODO: Should allow for different source and target embedders
        self.embedder  = embedder.to(device) if embedder is not None else embedder
        ### Dataloader Args - Maybe would be cleaner to use **kwargs here rather than in geomloss?
        ### or pass as dict: dataloader_args = {} etc
        self.shuffle   = shuffle
        self.nworkers  = nworkers
        self.pin_memory = pin_memory
        self.infloaders = infinite_loaders # If None, will only use infinte when actually useful. True or False overrides.
        self.persistw  = persistent_workers
        self.multiprc  = multiprocessing_context
        #self.extra_dl_args = kwargs
        self.seed = seed

        # ##### DEBUG
        # self.persistw=False
        # self.nbatches=2
        # self.maxbatch=1000
        # print(vars(self))
        # breakpoint()
        # ##### DEBUG

    def process_args(self, da, db):
        MaxB, minB = self.maxbatch, self.minbatch
        if isinstance(da, DataLoader) and isinstance(db, DataLoader):
            # TODO: what's a reliable way to get total # samples from dataloader?
            assert not self.nbatches == 'dynamic', "BatchifiedSamplesLoss can't do dynamic batch sizing if passing dataloaders"
            dla, dlb = da, db
            N_a, N_b = len(dla.dataset), len(dlb.dataset)
            NB_a, NB_b = len(dla), len(dlb)
            if self.nbatches == 'min':
                NB = min(NB_a, NB_b) #len(dla), len(dlb))
            elif self.nbatches == 'max':
                NB = max(NB_a, NB_b)
            elif type(self.nbatches) is int:
                NB = self.nbatches
        elif isinstance(da, Dataset) and isinstance(db, Dataset):
            N_a, N_b = len(da), len(db) # FIXME: does this always work?
            if self.nbatches == 'dynamic' and max(N_a/MaxB,N_b/MaxB) <= min(N_a/minB,N_b/minB):
                # Same number of batches, batch size determined by dataset size
                # Second condition guarantees that batch_size is in [minB, MaxB] for both datasets
                NB  = math.ceil(max(N_a/MaxB,N_b/MaxB))
                B_a = int(N_a/NB)
                B_b = int(N_b/NB)
            else:
                # Same batch size, number determined by dataset sizes or prespecified
                B_a = min(self.maxbatch, N_a) # Can't load a batch bigger than dataset!
                B_b = min(self.maxbatch, N_b) # Can't load a batch bigger than dataset!
                # Only include last batch if it's above minB threshold
                NB_a = int(N_a/B_a) + (N_a % B_a >= minB)
                NB_b = int(N_b/B_b) + (N_b % B_b >= minB)
                if self.nbatches == 'min':
                    NB = min(NB_a, NB_b) #len(dla), len(dlb))
                elif self.nbatches == 'max':
                    NB = max(NB_a, NB_b)
                elif type(self.nbatches) is int:
                    NB = self.nbatches
                else: # E.g. if nbatches=='dynamic' but second condition above was False
                    NB = max(NB_a, NB_b)

            ### Compute effective number of batches and batch sizes
            # We will drop the last batch if it's too small
            drop_last_a = (N_a % B_a > 0) and (N_a % B_a < minB)
            drop_last_b = (N_b % B_b > 0) and (N_b % B_b < minB)
            # Effective number of batches
            NB_a = int(N_a/B_a) + int(N_a % B_a > 0 and not drop_last_a)
            NB_b = int(N_b/B_b) + int(N_b % B_b > 0 and not drop_last_b)

            logger.warning(f'Dataset a: {NB_a} batches of size {B_a} (drop last: {drop_last_a}, recycle: {NB>NB_a})')
            logger.warning(f'Dataset b: {NB_b} batches of size {B_b} (drop last: {drop_last_b}, recycle: {NB>NB_b})')

            dl_args = {
                'shuffle': self.shuffle,
                'num_workers': self.nworkers,
                'pin_memory': self.pin_memory,
                'persistent_workers': self.persistw,
                'multiprocessing_context': self.multiprc,
            }
            print(dl_args)

            if self.seed:
                g = torch.Generator()
                g.manual_seed(self.seed)
                dl_args['worker_init_fn'] = seed_worker
                dl_args['generator'] = g
            # TODO: For self.nbatches == 'max , only need to persist / use InfiniteDataLoader
            # for dataset with less batches. The other one shouldn't need cycling, unless
            # explicitly required for repeated computation with same datasets.
            # Also true for both if self.nbatches == 'min'.
            if (not self.infloaders) or (self.infloaders is None and (NB_a == 1 or NB_a >= NB)):
                dla = DataLoader(da, batch_size=B_a, drop_last=drop_last_a, **dl_args)
            else:
                dla = InfiniteDataLoader(da, batch_size=B_a, drop_last=drop_last_a, **dl_args)

            if (not self.infloaders) or (self.infloaders is None and (NB_b == 1 or NB_b >= NB)):
                dlb = DataLoader(db, batch_size=B_b, drop_last=drop_last_b, **dl_args)
            else:
                dlb = InfiniteDataLoader(db, batch_size=B_b, drop_last=drop_last_b, **dl_args)

            #dla = InfiniteDataLoader(da, batch_size=B_a, drop_last=drop_last_a, **dl_args)
            #dlb = InfiniteDataLoader(db, batch_size=B_b, drop_last=drop_last_b, **dl_args)
        elif (type(da) is tuple) and torch.is_tensor(da[0]) and isinstance(db, DataLoader):
            # One of the datasets already loaded into memory, batchify the other only
            # Todo generalize for the case where db is a tensor instead
            dlb   = db
            Xa,Ya = da
            N_a, N_b = Xa.shape[0], len(dlb.dataset)
            NB_a, NB_b = 1, len(dlb)
            NB = NB_b # In this case we implcitly do self.nbatches == 'max'
            dla = [(Xa,Ya)]
        else:
            raise  ValueError("Incompatible data object")

        logger.info('Preprocessing done!')
        return dla, dlb, N_a, N_b, NB


    def forward(self, a, b):
        dla, dlb, N, M, K = self.process_args(a, b)
        NB_a, NB_b = len(dla), len(dlb)
        gena, genb = iter(dla), iter(dlb)
        total_loss = 0
        self.seen_samples = {'a': 0, 'b': 0}

        logger.info(f'Loader types: {dla.__class__.__name__} (src) / {dlb.__class__.__name__} (tgt)')

        def safe_next(itr, gen):
            """ Get next item from generator(iterable), restart if end is reached """
            try: item = next(gen)
            except StopIteration:
                logger.info('Reinitializing iterator...')
                gen  = iter(itr)
                item = next(gen)
            return item, gen

        def prepare_batch(batch, side=0):
            x,y = (batch[0], batch[1]) if len(batch) == 2 else (batch,None)
            B = x.shape[0]
            assert B >= self.minbatch, f'Batch size ({B}) below minbatch size ({self.minbatch})'
            x,y = x.to(self.device),y.to(self.device)
            if self.embedder:
                x = self.embedder(x)
            if self.on_joint: # flatten and concat labels on last dim
                if self.shift_ys[side] is not None:
                    y += self.shift_ys[side]
                z = torch.cat((x.view(B, -1), y.type(x.dtype).unsqueeze(1)), -1)
                # Geomloss expects batched data - unsqueeze on first dim - NOT TRUE ANYMORE
                #z = z.view(1, B, -1)
                z = z.view(B, -1)
            else: # just flatten
                #z = x.view(1, B, -1)
                z = x.view(B, -1)
            return z


        for k in tqdm(range(K), desc="Batchified samples loss", leave=False):
            # If either dataset has only one batch, just load once and keep in mem
            if (NB_a > 1) or (k == 0):
                if PROFILING: torch.cuda.synchronize()
                ta0 = time()
                batcha, gena = safe_next(dla, gena)
                if PROFILING: torch.cuda.synchronize()
                ta1 = time()
                za  = prepare_batch(batcha, side=0)
                if PROFILING: torch.cuda.synchronize()
                ta2 = time()
            else:
                ta2 = ta1 = ta0 = time()
                logger.info(f'Reusing already-prepared batch for source')

            if (NB_b > 1) or (k == 0):
                if PROFILING: torch.cuda.synchronize()
                tb0 = time()
                batchb, genb = safe_next(dlb, genb)
                if PROFILING: torch.cuda.synchronize()
                tb1 = time()
                zb  = prepare_batch(batchb, side=1)
                if PROFILING: torch.cuda.synchronize()
                tb2 = time()
            else:
                tb2 = tb1 = tb0 = time()
                logger.info(f'Reusing already-prepared batch for target')

            n, d = za.shape
            m, _ = zb.shape
            logger.info(f'Effective batch sizes received: {n}x{d-1},{m}x{d-1}')
            logger.info(f'Time to load batches: {ta1-ta0:8.4e}s {tb1-tb0:8.4e}s')
            logger.info(f'Time to prepare data: {ta2-ta1:8.4e}s {tb2-tb1:8.4e}s')

            #### Method 1: reduce weight on samples
            #α = torch.ones(b, n).type_as(x) / N # Divide by *total* nsamples
            #β = torch.ones(b, m).type_as(y) / M # Divide by *total* nsamples
            #batch_loss = self.sl(α, x, β, y)
            #### Method 2: treat as if they add up to one, then divide at the end
            # This seems to be what https://arxiv.org/pdf/1910.04091.pdf does
            if PROFILING: torch.cuda.synchronize()
            t3 = time()
            batch_loss = self.batchloss(za, zb)
            t4 = time()
            logger.info(f'Time to run SamplesLoss: {t4-t3:8.4e}s')

            total_loss += batch_loss
            # TODO: these sould account for possible repetitions
            self.seen_samples['a'] += n
            self.seen_samples['b'] += m

        # If method 2, still need to average:
        total_loss /= K

        return total_loss
