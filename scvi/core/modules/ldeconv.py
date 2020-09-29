# -*- coding: utf-8 -*-
"""Main module."""

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.distributions import Normal

from scvi.core._distributions import (
    NegativeBinomial, Poisson
)

from typing import Tuple, Dict

torch.backends.cudnn.benchmark = True


# linear model for single-cell data
class scDeconv(nn.Module):
    """
    Linear model for cell-type modeling of single-cell data.
    
    This is an re-implementation of the ScModel module of stereoscope "https://github.com/almaan/stereoscope/blob/master/stsc/models.py"

    Parameters
    ----------
    n_input
        Number of input genes
    n_labels
        Number of cell types
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    """

    def __init__(
        self,
        n_input: int,
        n_labels: int,
        gene_likelihood: str = "nb",
    ):
        super().__init__()
        self.gene_likelihood = gene_likelihood
        self.n_labels = n_labels
        self.n_genes = n_input

        #####
        #
        # x_ng \sim NB(l_n * softplus(W_{g, c_n}), exp(theta))
        #
        #####
        # dispersion for negative binomial
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        self.W = torch.nn.Parameter(torch.randn(n_input, n_labels)) # n_genes, n_cell types

    def get_weights(self, softplus=True) -> torch.Tensor:
        """
        Returns the (positive) weights W.

        Returns
        -------
        type
            tensor
        """
        res = self.W
        if softplus:
            res = torch.nn.functional.softplus(res)
        return res.detach().cpu().numpy()

    def get_dispersion(self, exp=True) -> torch.Tensor:
        """
        Returns the (positive) dispersion px_r.

        Returns
        -------
        type
            tensor
        """
        res = self.px_r
        if exp:
            res = torch.exp(res)
        return res.detach().cpu().numpy()

    def get_sample_scale(
        self, x, y, n_samples=1,
    ) -> torch.Tensor:
        """
        Returns the tensor of predicted frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of values with shape ``(batch_size, n_label)``
        n_samples
            number of samples (Default value = 1)

        Returns
        -------
        type
            tensor of predicted frequencies of expression with shape ``(batch_size, n_input)``
        """
        return self.inference(x, y, n_samples=n_samples)["px_scale"]

    def get_sample_rate(
        self, x, y, n_samples=1,
    ) -> torch.Tensor:
        """
        Returns the tensor of means of the negative binomial distribution.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y
            tensor of values with shape ``(batch_size, n_label)``
        n_samples
            number of samples (Default value = 1)

        Returns
        -------
        type
            tensor of means of the negative binomial distribution with shape ``(batch_size, n_input)``
        """
        return self.inference(x, y, n_samples=n_samples)["px_rate"]

    def get_reconstruction_loss(
        self, x, px_rate, px_r, **kwargs
    ) -> torch.Tensor:
        if self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    def inference(
        self, x, y,
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        # there are no latent variables to infer, gene expression only depends on y
        px_r = torch.exp(self.px_r)
        px_scale = torch.nn.functional.softplus(self.W)[:, y[:, 0]].T # cells per gene
        library = torch.sum(x, dim=1, keepdim=True)
        px_rate = library * px_scale

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            library=library,
        )

    def forward(
        self, x, y, ind_x
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the reconstruction loss and the KL divergences.

        Parameters
        ----------
        x
            tensor of values with shape (batch_size, n_input)
        y
            tensor of cell-types labels with shape (batch_size, n_labels)

        Returns
        -------
        type
            the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution
        outputs = self.inference(x, y)
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r)

        return reconst_loss, 0.0, 0.0


# deconvolution model for spatial transcriptomics data
class stDeconv(nn.Module):
    """
    Linear model for cell-type deconvolution of spatial transcriptomics data.

    This is an re-implementation of the STModel module of stereoscope "https://github.com/almaan/stereoscope/blob/master/stsc/models.py"

    Parameters
    ----------
    n_spots
        Number of input spots
    params 
        Tuple of ndarray of shapes [(n_genes, n_labels), (n_genes)] containing the dictionnary and log dispersion parameters
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    """

    def __init__(
        self,
        n_spots: int,
        params: Tuple[np.ndarray],
        gene_likelihood: str = "nb",
    ):
        super().__init__()
        self.gene_likelihood = gene_likelihood
        self.W = torch.tensor(params[0])
        self.px_r = torch.tensor(params[1])
        self.n_spots = n_spots
        self.n_genes, self.n_labels = self.W.shape

        #####
        #
        # x_sg \sim NB(\beta_g * \sum_{z=1}^Z exp(v_sz) * softplus(W)_gz + \gamma_s \eta_g, exp(px_r))
        #
        #####
        # note, the gamma is included in the V below!

        # noise from data
        self.eta = torch.nn.Parameter(torch.randn(self.n_genes))
        # factor loadings
        self.V = torch.nn.Parameter(torch.randn(self.n_labels + 1, self.n_spots))
        # additive gene bias
        self.beta = torch.nn.Parameter(torch.randn(self.n_genes))


    def get_loadings(self) -> torch.Tensor:
        """
        Returns the loadings V.

        Returns
        -------
        type
            tensor
        """
        return self.V.detach().cpu().numpy()

    def get_reconstruction_loss(
        self, x, px_rate, px_r, **kwargs
    ) -> torch.Tensor:
        if self.gene_likelihood == "nb":
            reconst_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    def inference(
        self, x, ind_x,
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        # x_sg \sim NB(softplus(\beta_g) * \sum_{z=1}^Z softplus(v_sz) * softplus(W)_gz + \gamma_s \eta_g, exp(px_r))

        px_r = torch.exp(self.px_r) # n_genes
        beta = torch.nn.functional.softplus(self.beta) # n_genes
        v = torch.nn.functional.softplus(self.V) # n_labels + 1, n_spots
        w = torch.nn.functional.softplus(self.W)  # n_genes, n_labels
        eps = torch.nn.functional.softplus(self.eta) # n_genes

        # account for gene specific bias and add noise
        r_hat = torch.cat([beta.unsqueeze(1) * w, eps.unsqueeze(1)], dim=1) # n_genes, n_labels + 1
        # subsample observations
        v_ind = v[:, ind_x[:, 0]] # labels + 1, batch_size
        px_rate = torch.transpose(torch.matmul(r_hat, v_ind), 0, 1) # batch_size, n_genes 

        return dict(
            px_r=px_r,
            px_rate=px_rate,
            eta=self.eta
        )

    def forward(
        self, x, y, ind_x,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the reconstruction loss and the KL divergences.

        Parameters
        ----------
        x
            tensor of values with shape (batch_size, n_input)
        ind_x
            tensor of indices with shape (batch_size,)

        Returns
        -------
        type
            the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution
        outputs = self.inference(x, ind_x)
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        eta = outputs["eta"]

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r)

        # prior likelihood
        mean = torch.zeros_like(eta)
        scale = torch.ones_like(eta)

        log_likelihood_prior = Normal(mean, scale).log_prob(eta).sum()
         
        return reconst_loss, 0.0, log_likelihood_prior

