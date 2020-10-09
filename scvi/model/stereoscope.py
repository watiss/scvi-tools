import logging

from anndata import AnnData
import numpy as np
from scvi._compat import Literal
from scvi.core.modules import scDeconv, stDeconv, stDeconvAmortized, stVAE
from scvi.core.models import BaseModelClass
from scvi.core.trainers import CustomStereoscopeTrainer
from scvi.core.data_loaders import CustomStereoscopeDataLoader, ScviDataLoader
from typing import Optional, List
from collections import OrderedDict

import torch
from scvi._constants import _CONSTANTS

logger = logging.getLogger(__name__)


class scStereoscope(BaseModelClass):
    """
    Reimplementation of Stereoscope for deconvolution of spatial transcriptomics from single-cell transcriptomics.
    
    https://github.com/almaan/stereoscope.

    Parameters
    ----------
    sc_adata
        single-cell AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    use_cuda
        Use the GPU or not.
    **model_kwargs
        Keyword args for :class:`~scvi.core.modules.VAE`

    Examples
    --------
    >>> sc_adata = anndata.read_h5ad(path_to_sc_anndata)
    >>> scvi.data.setup_anndata(sc_adata, label_key="labels")
    >>> stereo = scvi.model.scStereoscope(sc_adata)
    >>> stereo.train()
    >>> weights = stereo.get_weights()
    """

    def __init__(
        self,
        sc_adata: AnnData,
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        use_cuda: bool = True,
        **model_kwargs,
    ):
        super(scStereoscope, self).__init__(sc_adata, use_cuda=use_cuda)

        # first we have the scRNA-seq model
        self.model = scDeconv(
            n_input=self.summary_stats["n_genes"],
            n_labels=self.summary_stats["n_labels"],
            gene_likelihood=gene_likelihood,
            **model_kwargs,
        )
        self._model_summary_string = (
            "scDeconv Model with params: \ngene_likelihood: {}"
        ).format(
            gene_likelihood,
        )
        self.init_params_ = self._get_init_params(locals())

    def get_params(self):
        return self.model.get_weights(softplus=False), self.model.get_dispersion()

    def train(self,
        n_epochs: Optional[int] = None,
        train_size: float = 0.9,
        test_size: Optional[float] = None,
        lr: float = 1e-3,
        frequency: Optional[int] = None,
        train_fun_kwargs: dict = {},
        **kwargs,
    ):
        train_fun_kwargs = dict(train_fun_kwargs)
        if self.is_trained_ is False:
            self.trainer = self._trainer_class(
                self.model,
                self.adata,
                train_size=train_size,
                test_size=test_size,
                frequency=frequency,
                use_cuda=self.use_cuda,
                **kwargs,
            )
            self.train_indices_ = self.trainer.train_set.indices
            self.test_indices_ = self.trainer.test_set.indices
            self.validation_indices_ = self.trainer.validation_set.indices
        # for autotune
        if "n_epochs" not in train_fun_kwargs:
            if n_epochs is None:
                n_cells = self.adata.n_obs
                n_epochs = np.min([round((20000 / n_cells) * 400), 400])
            train_fun_kwargs["n_epochs"] = n_epochs
        if "lr" not in train_fun_kwargs:
            train_fun_kwargs["lr"] = lr
        self.trainer.train(**train_fun_kwargs)
        self.is_trained_ = True

    @property
    def _trainer_class(self):
        return CustomStereoscopeTrainer

    @property
    def _scvi_dl_class(self):
        return ScviDataLoader


class stStereoscope(BaseModelClass):
    """
    Reimplementation of Stereoscope for deconvolution of spatial transcriptomics from single-cell transcriptomics.
    
    https://github.com/almaan/stereoscope.

    Parameters
    ----------
    st_adata
        spatial transcriptomics AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    weights
        weights from scStereoscope model 
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    use_cuda
        Use the GPU or not.
    **model_kwargs
        Keyword args for :class:`~scvi.core.modules.VAE`

    Examples
    --------
    >>> st_adata = anndata.read_h5ad(path_to_st_anndata)
    >>> scvi.data.setup_anndata(st_adata)
    >>> stereo = scvi.model.stStereoscope(sc_adata)
    >>> stereo.train()
    >>> st_adata.obs["deconv"] = stereo.get_deconvolution()
    """

    def __init__(
        self,
        st_adata: AnnData,
        params: np.ndarray,
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        amortized: bool = False,
        use_cuda: bool = True,
        **model_kwargs,
    ):
        super(stStereoscope, self).__init__(st_adata, use_cuda=use_cuda)

        self.amortized = amortized
        # first we have the scRNA-seq model
        class_ = stDeconv
        if self.amortized:
            class_ = stDeconvAmortized
        self.model = class_(
            n_spots=st_adata.n_obs,
            params=params,
            gene_likelihood=gene_likelihood,
            use_cuda=True,
            **model_kwargs,
        )
        self._model_summary_string = (
            "stDeconv Model with params: \ngene_likelihood: {}"
        ).format(
            gene_likelihood,
        )
        self.init_params_ = self._get_init_params(locals())


    @torch.no_grad()
    def get_proportions(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the proportions for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.amortized is False:
            return self.model.get_proportions()

        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        scdl = self._make_scvi_dl(adata=adata, batch_size=batch_size)
        latent = []
        for tensors in scdl:
            x = tensors[_CONSTANTS.X_KEY]
            z = self.model.get_proportions(x)
            latent += [z.cpu()]
        return np.array(torch.cat(latent))

    def train(self,
        n_epochs: Optional[int] = None,
        train_size: float = 0.9,
        test_size: Optional[float] = None,
        lr: float = 1e-3,
        frequency: Optional[int] = None,
        train_fun_kwargs: dict = {},
        **kwargs,
    ):
        train_fun_kwargs = dict(train_fun_kwargs)
        if self.is_trained_ is False:
            self.trainer = self._trainer_class(
                self.model,
                self.adata,
                train_size=train_size,
                test_size=test_size,
                frequency=frequency,
                use_cuda=self.use_cuda,
                **kwargs,
            )
            self.train_indices_ = self.trainer.train_set.indices
            self.test_indices_ = self.trainer.test_set.indices
            self.validation_indices_ = self.trainer.validation_set.indices
        # for autotune
        if "n_epochs" not in train_fun_kwargs:
            if n_epochs is None:
                n_cells = self.adata.n_obs
                n_epochs = np.min([round((20000 / n_cells) * 400), 400])
            train_fun_kwargs["n_epochs"] = n_epochs
        if "lr" not in train_fun_kwargs:
            train_fun_kwargs["lr"] = lr
        self.trainer.train(**train_fun_kwargs)
        self.is_trained_ = True

    @property
    def _trainer_class(self):
        return CustomStereoscopeTrainer

    @property
    def _scvi_dl_class(self):
        return CustomStereoscopeDataLoader


class stVI(BaseModelClass):
    """
    Model for continuous deconvolution.    

    Parameters
    ----------
    st_adata
        spatial transcriptomics AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    state_dict
        state_dict from CondSCVI model 
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    use_cuda
        Use the GPU or not.
    **model_kwargs
        Keyword args for :class:`~scvi.core.modules.VAE`

    Examples
    --------
    >>> st_adata = anndata.read_h5ad(path_to_st_anndata)
    >>> scvi.data.setup_anndata(st_adata)
    >>> stereo = scvi.model.stStereoscope(sc_adata)
    >>> stereo.train()
    >>> st_adata.obs["deconv"] = stereo.get_deconvolution()
    """

    def __init__(
        self,
        st_adata: AnnData,
        n_labels: int,
        state_dict: List[OrderedDict],
        gene_likelihood: Literal["nb", "poisson"] = "nb",
        use_cuda: bool = True,
        **model_kwargs,
    ):
        super(stVI, self).__init__(st_adata, use_cuda=use_cuda)

        # first we have the scRNA-seq model
        self.model = stVAE(
            n_labels=n_labels,
            n_genes=st_adata.n_vars,
            state_dict=state_dict,
            use_cuda=True,
            **model_kwargs,
        )
        self._model_summary_string = (
            "stVI Model with params: \ngene_likelihood: {}"
        ).format(
            gene_likelihood,
        )
        self.init_params_ = self._get_init_params(locals())


    @torch.no_grad()
    def get_proportions(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the proportions for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        scdl = self._make_scvi_dl(adata=adata, batch_size=batch_size)
        prop = []
        gamma = []
        for tensors in scdl:
            x = tensors[_CONSTANTS.X_KEY]

            v = self.model.get_proportions(x)
            g = self.model.get_gamma(x)

            prop += [v.cpu()]
            gamma += [g.cpu()]
        return np.array(torch.cat(prop)), np.array(torch.cat(gamma))

    def train(self,
        n_epochs: Optional[int] = None,
        train_size: float = 0.9,
        test_size: Optional[float] = None,
        lr: float = 1e-3,
        frequency: Optional[int] = None,
        train_fun_kwargs: dict = {},
        **kwargs,
    ):
        train_fun_kwargs = dict(train_fun_kwargs)
        if self.is_trained_ is False:
            self.trainer = self._trainer_class(
                self.model,
                self.adata,
                train_size=train_size,
                test_size=test_size,
                frequency=frequency,
                use_cuda=self.use_cuda,
                **kwargs,
            )
            self.train_indices_ = self.trainer.train_set.indices
            self.test_indices_ = self.trainer.test_set.indices
            self.validation_indices_ = self.trainer.validation_set.indices
        # for autotune
        if "n_epochs" not in train_fun_kwargs:
            if n_epochs is None:
                n_cells = self.adata.n_obs
                n_epochs = np.min([round((20000 / n_cells) * 400), 400])
            train_fun_kwargs["n_epochs"] = n_epochs
        if "lr" not in train_fun_kwargs:
            train_fun_kwargs["lr"] = lr
        self.trainer.train(**train_fun_kwargs)
        self.is_trained_ = True

    @property
    def _trainer_class(self):
        return CustomStereoscopeTrainer

    @property
    def _scvi_dl_class(self):
        return CustomStereoscopeDataLoader


