# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches,too-many-instance-attributes,attribute-defined-outside-init,redefined-outer-name,unbalanced-tuple-unpacking,inconsistent-return-statements
from ot.da import joint_OT_mapping_kernel, joint_OT_mapping_linear
from ot.utils import BaseEstimator, check_params, kernel


class MappingTransport(BaseEstimator):

    """MappingTransport: DA methods that aims at jointly estimating a optimal
    transport coupling and the associated mapping

    Parameters
    ----------
    mu : float, optional (default=1)
        Weight for the linear OT loss (>0)
    eta : float, optional (default=0.001)
        Regularization term for the linear mapping `L` (>0)
    bias : bool, optional (default=False)
        Estimate linear mapping with constant bias
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    kernel : string, optional (default="linear")
        The kernel to use either linear or gaussian
    sigma : float, optional (default=1)
        The gaussian kernel parameter
    max_iter : int, optional (default=100)
        Max number of BCD iterations
    tol : float, optional (default=1e-5)
        Stop threshold on relative loss decrease (>0)
    max_inner_iter : int, optional (default=10)
        Max number of iterations (inner CG solver)
    inner_tol : float, optional (default=1e-6)
        Stop threshold on error (inner CG solver) (>0)
    log : bool, optional (default=False)
        record log if True
    verbose : bool, optional (default=False)
        Print information along iterations
    verbose2 : bool, optional (default=False)
        Print information along iterations

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    mapping_ :
        The associated mapping

        - array-like, shape (`n_features` (+ 1), `n_features`),
          (if bias) for kernel == linear

        - array-like, shape (`n_source_samples` (+ 1), `n_features`),
          (if bias) for kernel == gaussian
    log_ : dictionary
        The dictionary of log, empty dict if parameter log is not True


    References
    ----------
    .. [8] M. Perrot, N. Courty, R. Flamary, A. Habrard,
            "Mapping estimation for discrete optimal transport",
            Neural Information Processing Systems (NIPS), 2016.

    """

    def __init__(
        self,
        mu=1,
        eta=0.001,
        bias=False,
        metric="sqeuclidean",
        norm=None,
        kernel="linear",
        sigma=1,
        max_iter=100,
        tol=1e-5,
        max_inner_iter=10,
        inner_tol=1e-6,
        log=False,
        verbose=False,
        verbose2=False,
    ):
        self.metric = metric
        self.norm = norm
        self.mu = mu
        self.eta = eta
        self.bias = bias
        self.kernel = kernel
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.log = log
        self.verbose = verbose
        self.verbose2 = verbose2

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        r"""Builds an optimal coupling and estimates the associated mapping
        from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self
        """
        self._get_backend(Xs, ys, Xt, yt)

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            self.xs_ = Xs
            self.xt_ = Xt

            if self.kernel == "linear":
                returned_ = joint_OT_mapping_linear(
                    Xs,
                    Xt,
                    mu=self.mu,
                    eta=self.eta,
                    bias=self.bias,
                    verbose=self.verbose,
                    numItermax=self.max_iter,
                    numInnerItermax=self.max_inner_iter,
                    stopThr=self.tol,
                    stopInnerThr=self.inner_tol,
                    log=self.log,
                )

            elif self.kernel == "gaussian":
                returned_ = joint_OT_mapping_kernel(
                    Xs,
                    Xt,
                    mu=self.mu,
                    eta=self.eta,
                    bias=self.bias,
                    sigma=self.sigma,
                    verbose=self.verbose,
                    numItermax=self.max_iter,
                    numInnerItermax=self.max_inner_iter,
                    stopInnerThr=self.inner_tol,
                    stopThr=self.tol,
                    log=self.log,
                )

            # deal with the value of log
            if self.log:
                self.coupling_, self.mapping_, self.log_ = returned_
            else:
                self.coupling_, self.mapping_ = returned_
                self.log_ = {}

        return self

    def transform(self, Xs):
        r"""Transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if nx.array_equal(self.xs_, Xs):
                # perform standard barycentric mapping
                transp = self.coupling_ / nx.sum(self.coupling_, 1)[:, None]

                # set nans to 0
                transp[~nx.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = nx.dot(transp, self.xt_)
                return transp_Xs, transp
            else:
                if self.kernel == "gaussian":
                    K = kernel(Xs, self.xs_, method=self.kernel, sigma=self.sigma)
                elif self.kernel == "linear":
                    K = Xs
                if self.bias:
                    K = nx.concatenate(
                        [K, nx.ones((Xs.shape[0], 1), type_as=K)], axis=1
                    )
                transp_Xs = nx.dot(K, self.mapping_)

                return transp_Xs, None
