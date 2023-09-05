# -*- coding: utf-8 -*-
"""
    Routines for computing matrix square roots.

    With ideas from:

    https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
    https://github.com/pytorch/pytorch/issues/25481
"""

import pdb
import torch
from torch.autograd import Function
from functools import partial
import numpy as np
import scipy.linalg
try:
    import cupy as cp
except:
    import numpy as cp

#### VIA SVD, version 1: from https://github.com/pytorch/pytorch/issues/25481
def symsqrt_v1(A, func='symeig'):
    """Compute the square root of a symmetric positive definite matrix."""
    ## https://github.com/pytorch/pytorch/issues/25481#issuecomment-576493693
    ## perform the decomposition
    ## Recall that for Sym Real matrices, SVD, EVD coincide, |λ_i| = σ_i, so
    ## for PSD matrices, these are equal and coincide, so we can use either.
    if func == 'symeig':
        # Deprecated: s, v = A.symeig(eigenvectors=True) # This is faster in GPU than CPU, fails gradcheck. See https://github.com/pytorch/pytorch/issues/30578
        s,v = torch.linalg.eigh(A, UPLO='U')
    elif func == 'svd':
        _, s, v = A.svd()                 # But this passes torch.autograd.gradcheck()
    else:
        raise ValueError()

    ## truncate small components
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


#### VIA SVD, version 2: from https://github.com/pytorch/pytorch/issues/25481
def symsqrt_v2(A, func='symeig'):
    """Compute the square root of a symmetric positive definite matrix."""
    # perform the decomposition
    if func == 'symeig':
        # DEPRECATED: s, v = A.symeig(eigenvectors=True) # This is faster in GPU than CPU, fails gradcheck. See https://github.com/pytorch/pytorch/issues/30578
        s,v = torch.linalg.eigh(A, UPLO='U')
    elif func == 'svd':
        _, s, v = A.svd()                 # But this passes torch.autograd.gradcheck()
    else:
        raise ValueError()

    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps

    ### This doesn't work for batched version
    # s = s[..., above_cutoff]
    # v = v[..., above_cutoff] # Doesn't work for batched version
    # sol = (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

    ### This does but fails gradcheck because of inpalce
    #s[...,~above_cutoff] = 0

    ### This seems to be equivalent to above, work for batch, and pass inplace. CHECK!!!!
    s = torch.where(above_cutoff, s, torch.zeros_like(s))

    sol =torch.matmul(torch.matmul(v,torch.diag_embed(s.sqrt(),dim1=-2,dim2=-1)),v.transpose(-2,-1))

    return sol

# ########### VIA SVD: from https://github.com/pytorch/pytorch/issues/25481
# def symsqrt_diff(A):
#     """Compute the square root of a symmetric positive definite matrix.
#         Seems to work for batched version. Check.
#     """
#     # perform the decomposition
#     _, s, v = A.svd()                 # But this passes torch.autograd.gradcheck()
#
#     # truncate small components
#     above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
#     s = torch.where(above_cutoff, s, torch.zeros_like(s))
#     sol =torch.matmul(torch.matmul(v,torch.diag_embed(s.sqrt(),dim1=-2,dim2=-1)),v.transpose(-2,-1))
#
#     return sol

def special_sylvester(a, b):
    """Solves the eqation `A @ X + X @ A = B` for a positive definite `A`."""
    # https://math.stackexchange.com/a/820313
    # DEPRECATED: s, v = a.symeig(eigenvectors=True)
    s,v = torch.linalg.eigh(a, UPLO='U')
    d = s.unsqueeze(-1)
    d = d + d.transpose(-2, -1)
    vt = v.transpose(-2, -1)
    c = vt @ b @ v
    return v @ (c / d) @ vt


##### Via Newton-Schulz: based on
## https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py, and
## https://github.com/BorisMuzellec/EllipticalEmbeddings/blob/master/utils.py
def sqrtm_newton_schulz(A, numIters, reg=None, return_error=False, return_inverse=False):
    """ Matrix squareroot based on Newton-Schulz method """
    if A.ndim <= 2: # Non-batched mode
        A = A.unsqueeze(0)
        batched = False
    else:
        batched = True
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = (A**2).sum((-2,-1)).sqrt() # Slightly faster than : A.mul(A).sum((-2,-1)).sqrt()

    if reg:
        ## Renormalize so that the each matrix has a norm lesser than 1/reg,
        ## but only normalize when necessary
        normA *= reg
        renorm = torch.ones_like(normA)
        renorm[torch.where(normA > 1.0)] = normA[cp.where(normA > 1.0)]
    else:
        renorm = normA

    Y = A.div(renorm.view(batchSize, 1, 1).expand_as(A))
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).to(A.device)#.type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).to(A.device)#.type(dtype)
    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA    = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    sAinv = Z/torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    if not batched:
        sA = sA[0,:,:]
        sAinv = sAinv[0,:,:]

    if not return_inverse and not return_error:
        return sA
    elif not return_inverse and return_error:
        return sA, compute_error(A, sA)
    elif return_inverse and not return_error:
        return sA,sAinv
    else:
        return sA, sAinv, compute_error(A, sA)

def create_symm_matrix(batchSize, dim, numPts=20, tau=1.0, dtype=torch.float32,
    verbose=False):
    """ Creates a random PSD matrix """
    A = torch.zeros(batchSize, dim, dim).type(dtype)
    for i in range(batchSize):
        pts = np.random.randn(numPts, dim).astype(np.float32)
        sA = np.dot(pts.T, pts)/numPts + tau*np.eye(dim).astype(np.float32);
        A[i,:,:] = torch.from_numpy(sA);
    if verbose: print('Creating batch %d, dim %d, pts %d, tau %f, dtype %s' % (batchSize, dim, numPts, tau, dtype))
    return A

def compute_error(A, sA):
    """ Computes error in approximation """
    normA = torch.sqrt(torch.sum(torch.sum(A * A, dim=1),dim=1))
    error = A - torch.bmm(sA, sA)
    error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
    return torch.mean(error)

###==========================
# ## ANOTHER NS VERSION (Muzzelec and Cuturi)
# def batch_sqrtm(A, numIters = 20, reg = 2.0):
#     """
#     Batch matrix root via Newton-Schulz iterations
#     """
#     batchSize = A.shape[0]
#     dim = A.shape[1]
#     #Renormalize so that the each matrix has a norm lesser than 1/reg, but only normalize when necessary
#     normA = reg * cp.linalg.norm(A, axis=(1, 2))
#     renorm_factor = cp.ones_like(normA)
#     renorm_factor[cp.where(normA > 1.0)] = normA[cp.where(normA > 1.0)]
#     renorm_factor = renorm_factor.reshape(batchSize, 1, 1)
#     Y = cp.divide(A, renorm_factor)
#     I = cp.eye(dim).reshape(1, dim, dim).repeat(batchSize, axis=0)
#     Z = cp.eye(dim).reshape(1, dim, dim).repeat(batchSize, axis=0)
#     for i in range(numIters):
#         T = 0.5 * (3.0 * I - cp.matmul(Z, Y))
#         Y = cp.matmul(Y, T)
#         Z = cp.matmul(T, Z)
#     sA = Y * cp.sqrt(renorm_factor)
#     sAinv = Z / cp.sqrt(renorm_factor)
#     return sA, sAinv

# TODO: Make batch version of this
class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: square root is not differentiable for matrices with zero eigenvalues.

    """
    @staticmethod
    def forward(ctx, input, method = 'numpy'):
        _dev = input.device
        if method == 'numpy':
            # Via numpy: faster for small matrices in cpu
            m = input.cpu().detach().numpy().astype(np.float_)
            sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).type_as(input)
        elif method == 'pytorch':
            # Via pure pytorch: faster for large matrices in GPU
            sqrtm = symsqrt(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output, method = 'numpy'):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            if method == 'numpy':
                sqrtm = sqrtm.data.numpy().astype(np.float_)
                gm = grad_output.data.numpy().astype(np.float_)
                # Given a positive semi-definite matrix X,
                # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
                # matrix square root dX^{1/2} by solving the Sylvester equation:
                # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
                grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
                grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
            elif method == 'pytorch':
                grad_input = special_sylvester(sqrtm, grad_output)
        return grad_input


## ========================================================================== ##
## NOTE: Must pick which version of matrix square root to use!!!!
# I used
#  - symsqrt_v2 UP TO AND INCLUDING ICML SUMBISSION
#  - symsqrt_v1 USED JUST BEFORE NEURIPS
#  - symsqrt_diff # not necessary now that symeig passes grad check in bleeding edge pytorch

## sqrtm = MatrixSquareRoot.apply
sqrtm = symsqrt_v2
## sqrtm = symsqrt_v1
## sqrtm = symsqrt_diff
## ========================================================================== ##

def main():
    from torch.autograd import gradcheck

    k = torch.randn(5, 20, 20).double()
    M = k @ k.transpose(-1,-2)


    def pass_or_fail(b): print('PASS' if b else 'FAIL')

    ## Check symeig (now, linalg.eigh due to deprecation) itself
    print('Testing gradients of torch.linalg.eigh ...',end='')
    S = torch.rand(5,20,20, requires_grad=True).double()
    def func(S):
        x = 0.5 * (S + S.transpose(-2, -1))
        return torch.linalg.eigh(x)
    pass_or_fail(gradcheck(func, [S]))

    for method, func in [('symsqrt_v1', symsqrt_v1), ('symsqrt_v2', symsqrt_v2)]:
        print(f"*** Testing {method} ***")

        ### Test outputs
        print(f'Testing output of {method} via symeig ...',end='')
        s1 = func(M, func='symeig')
        test = torch.allclose(M, s1 @ s1.transpose(-1,-2))
        pass_or_fail(test)

        print(f'Testing output of {method} via svd ...',end='')
        s2 = func(M, func='svd')
        test = torch.allclose(M, s2 @ s2.transpose(-1,-2))
        pass_or_fail(test)

        print(f'Outputs of {method} with symeig and svd match ...', end='')
        pass_or_fail(torch.allclose(s1,s2))

        M.requires_grad = True

        ## Test gradients
        print(f'Testing gradients of {method} via symeig ...',end='')
        try:
            _sqrt = partial(func, func='symeig')
            test = gradcheck(_sqrt, (M,))
        except:
            test = False
        pass_or_fail(test)            

        print(f'Testing gradients of {method} via svd ...',end='')
        try:
            _sqrt = partial(func, func='svd')
            test = gradcheck(_sqrt, (M,))
        except:
            test = False
        pass_or_fail(test)




if __name__ == '__main__':
    main()
