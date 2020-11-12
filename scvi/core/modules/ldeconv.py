# -*- coding: utf-8 -*-
"""Main module."""

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.distributions import Normal

from torch.distributions import Normal, Categorical, kl_divergence as kl


from scvi.core._distributions import (
    NegativeBinomial, Poisson
)
from typing import Optional, List
from collections import OrderedDict

from scvi.core._log_likelihood import log_nb_positive_altparam
from scvi.core.modules._base import FCLayers

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
        # logit param for negative binomial
        self.px_o = torch.nn.Parameter(torch.randn(n_input))
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

    def get_dispersion(self) -> torch.Tensor:
        """
        Returns the dispersion px_o.

        Returns
        -------
        type
            tensor
        """
        res = self.px_o
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
        self, x, px_rate, px_o, **kwargs
    ) -> torch.Tensor:
        if self.gene_likelihood == "nb":
            reconst_loss = (
                -log_nb_positive_altparam(x, px_rate, px_o).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    def inference(
        self, x, y,
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        # there are no latent variables to infer, gene expression only depends on y
        px_scale = torch.nn.functional.softplus(self.W)[:, y[:, 0]].T # cells per gene
        library = torch.sum(x, dim=1, keepdim=True)
        px_rate = library * px_scale

        return dict(
            px_scale=px_scale,
            px_o=self.px_o,
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
        ind_x 
            tensor of indices (ignored)

        Returns
        -------
        type
            the reconstruction loss and the Kullback divergences
        """
        # Parameters for z latent distribution
        outputs = self.inference(x, y)
        px_rate = outputs["px_rate"]
        px_o = outputs["px_o"]

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_o)

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
        use_cuda=True
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.gene_likelihood = gene_likelihood
        # import parameters from sc model
        self.W = torch.tensor(params[0])
        self.px_o = torch.tensor(params[1])
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
        self.beta = torch.nn.Parameter(0.1 * torch.randn(self.n_genes))

    def get_proportions(self, keep_noise=False) -> torch.Tensor:
        """
        Returns the loadings.

        Returns
        -------
        type
            tensor
        """
        # get estimated unadjusted proportions
        res  = torch.nn.functional.softplus(self.V).detach().cpu().numpy().T # n_spots, n_labels + 1
        # remove dummy cell type proportion values
        if not keep_noise:
            res = res[:,:-1]
        # normalize to obtain adjusted proportions
        res = res / res.sum(axis = 1).reshape(-1,1)
        return res

    def get_reconstruction_loss(
        self, x, px_rate, px_o, **kwargs
    ) -> torch.Tensor:
        if self.gene_likelihood == "nb":
            reconst_loss = (
                -log_nb_positive_altparam(x, px_rate, px_o).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    def inference(
        self, x, ind_x,
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        # x_sg \sim NB(softplus(\beta_g) * \sum_{z=1}^Z softplus(v_sz) * softplus(W)_gz + \gamma_s \eta_g, exp(px_r))

        beta = torch.nn.functional.softplus(self.beta) # n_genes
        v = torch.nn.functional.softplus(self.V) # n_labels + 1, n_spots
        w = torch.nn.functional.softplus(self.W)  # n_genes, n_labels
        eps = torch.nn.functional.softplus(self.eta) # n_genes

        if self.use_cuda:
            w = w.cuda()
            px_o = self.px_o.cuda()
            eps = eps.cuda()
            v = v.cuda()
            beta = beta.cuda()

        # account for gene specific bias and add noise
        r_hat = torch.cat([beta.unsqueeze(1) * w, eps.unsqueeze(1)], dim=1) # n_genes, n_labels + 1
        # subsample observations
        v_ind = v[:, ind_x[:, 0]] # labels + 1, batch_size
        px_rate = torch.transpose(torch.matmul(r_hat, v_ind), 0, 1) # batch_size, n_genes 

        return dict(
            px_o=px_o,
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
        y
            tensor of cell types (ignored)
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
        px_o = outputs["px_o"]
        eta = outputs["eta"]

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_o)

        # prior likelihood
        mean = torch.zeros_like(eta)
        scale = torch.ones_like(eta)

        neg_log_likelihood_prior = -Normal(mean, scale).log_prob(eta).sum()
         
        return reconst_loss, 0.0, neg_log_likelihood_prior

class stDeconvAmortized(nn.Module):
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
        n_hidden: int=128,
        gene_likelihood: str = "nb",
        use_cuda=True
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.gene_likelihood = gene_likelihood
        # import parameters from sc model
        self.W = torch.tensor(params[0])
        self.px_o = torch.tensor(params[1])
        # self.n_spots = n_spots
        self.n_genes, self.n_labels = self.W.shape

        #####
        #
        # x_sg \sim NB(\beta_g * \sum_{z=1}^Z exp(v_sz) * softplus(W)_gz + \gamma_s \eta_g, exp(px_r))
        #
        #####
        # note, the gamma is included in the V below!

        # noise from data
        self.eta = torch.nn.Parameter(torch.randn(self.n_genes))
        # factor loadings via encoding
        self.encoder_V = FCLayers(
            n_in=self.n_genes,
            n_out=self.n_labels + 1,
            n_layers=2,
            n_hidden=n_hidden,
            dropout_rate=0.1,
        )
        # additive gene bias
        self.beta = torch.nn.Parameter(0.1 * torch.randn(self.n_genes))

    def get_proportions(self, x, keep_noise=False) -> torch.Tensor:
        """
        Returns the loadings.

        Returns
        -------
        type
            tensor
        """
        # get estimated unadjusted proportions
        res  = torch.nn.functional.softplus(self.encoder_V(x))
        # remove dummy cell type proportion values
        if not keep_noise:
            res = res[:,:-1]
        # normalize to obtain adjusted proportions
        res = res / res.sum(axis = 1).reshape(-1,1)
        return res

    def get_reconstruction_loss(
        self, x, px_rate, px_o, **kwargs
    ) -> torch.Tensor:
        if self.gene_likelihood == "nb":
            reconst_loss = (
                -log_nb_positive_altparam(x, px_rate, px_o).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    def inference(
        self, x, ind_x,
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        # x_sg \sim NB(softplus(\beta_g) * \sum_{z=1}^Z softplus(v_sz) * softplus(W)_gz + \gamma_s \eta_g, exp(px_r))

        beta = torch.nn.functional.softplus(self.beta) # n_genes
        v = self.encoder_V(x)
        v = torch.nn.functional.softplus(v) # minibatch_size, n_labels + 1
        w = torch.nn.functional.softplus(self.W)  # n_genes, n_labels
        eps = torch.nn.functional.softplus(self.eta) # n_genes

        if self.use_cuda:
            w = w.cuda()
            px_o = self.px_o.cuda()
            eps = eps.cuda()
            v = v.cuda()
            beta = beta.cuda()

        # account for gene specific bias and add noise
        r_hat = torch.cat([beta.unsqueeze(1) * w, eps.unsqueeze(1)], dim=1) # n_genes, n_labels + 1
        # subsample observations
        px_rate = torch.matmul(v, torch.transpose(r_hat, 0, 1)) # batch_size, n_genes 

        return dict(
            px_o=px_o,
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
        y
            tensor of cell types (ignored)
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
        px_o = outputs["px_o"]
        eta = outputs["eta"]

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_o)

        # prior likelihood
        mean = torch.zeros_like(eta)
        scale = torch.ones_like(eta)

        neg_log_likelihood_prior = -Normal(mean, scale).log_prob(eta).sum()
         
        return reconst_loss, 0.0, neg_log_likelihood_prior


class stVAE(nn.Module):
    """
    non-linear model for cell-type deconvolution of spatial transcriptomics data.

    Parameters
    ----------
    state_dict 
        state_dict of the decoder of the CondScVI
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    """

    def __init__(
        self,
        n_labels: int,
        state_dict: List[OrderedDict],
        n_genes:int,
        cell_type_prior: np.ndarray,
        n_latent: int=2,
        n_hidden: int=128,
        n_layers: int=1,
        gene_likelihood: str = "nb",
        use_cuda=True
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.gene_likelihood = gene_likelihood
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_labels = n_labels
        self.cell_type_prior = torch.nn.Parameter(
            cell_type_prior
            if cell_type_prior is not None
            else (1 / (n_labels+1)) * torch.ones(1, n_labels+1),
            requires_grad=False,
        )
        # import parameters from sc model
        self.decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_cat_list=[n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0.1,
        )
        self.px_decoder = nn.Sequential(
        nn.Linear(n_hidden, n_genes),
        nn.Softplus()
        )

        self.decoder.load_state_dict(state_dict[0])
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.px_decoder.load_state_dict(state_dict[1])
        for param in self.px_decoder.parameters():
            param.requires_grad = False
        self.px_o = torch.tensor(state_dict[2])

        # create encoders (for proportions)
        self.encoder= FCLayers(
            n_in=n_genes,
            n_out=n_hidden,
            n_cat_list=[],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0.1,
        )
        self.z_encoder = nn.Linear(n_hidden, n_labels + 1)

        # create separate encoder for gammas (avoid parameter sharing)?
        self.gamma_encoder = nn.Sequential(nn.Linear(n_hidden, n_latent * n_labels), nn.Tanh())
        self.transform_gamma = lambda x: 2 * x
        #####
        #
        # x_sg \sim NB(\beta_g * \sum_{z=1}^Z exp(v_sz) * softplus(W)_gz + \gamma_s \eta_g, exp(px_r))
        #
        #####
        # note, the gamma is included in the V below!

        # noise from data
        self.eta = torch.nn.Parameter(torch.randn(self.n_genes))
        # additive gene bias
        self.beta = torch.nn.Parameter(0.1 * torch.randn(self.n_genes))

    def get_proportions(self, x, keep_noise=False) -> torch.Tensor:
        """
        Returns the loadings.

        Returns
        -------
        type
            tensor
        """
        # get estimated unadjusted proportions
        h = self.encoder(x)
        v = self.z_encoder(h)
        res = torch.nn.functional.softplus(v)
        # remove dummy cell type proportion values
        if not keep_noise:
            res = res[:,:-1]
        # normalize to obtain adjusted proportions
        res = res / res.sum(axis = 1).reshape(-1,1)
        return res

    def get_gamma(self, x) -> torch.Tensor:
        """
        Returns the loadings.

        Returns
        -------
        type
            tensor
        """
        # get estimated unadjusted proportions
        h = self.encoder(x)
        return self.transform_gamma(self.gamma_encoder(h))

    def get_reconstruction_loss(
        self, x, px_rate, px_o, **kwargs
    ) -> torch.Tensor:
        if self.gene_likelihood == "nb":
            reconst_loss = (
                -log_nb_positive_altparam(x, px_rate, px_o).sum(dim=-1)
            )
        elif self.gene_likelihood == "poisson":
            reconst_loss = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return reconst_loss

    def inference(
        self, x, ind_x,
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        # x_sg \sim NB(softplus(\beta_g) * \sum_{z=1}^Z softplus(v_sz) * softplus(W)_gz + \gamma_s \eta_g, exp(px_r))
        M = x.shape[0]
        beta = torch.nn.functional.softplus(self.beta) # n_genes
        h = self.encoder(x)

        # first get the cell type proportion
        v = self.z_encoder(h)
        v = torch.nn.functional.softplus(v) # minibatch_size, n_labels + 1

        # second get the gamma values,  and the gene expression for all cell types
        gamma = self.transform_gamma(self.gamma_encoder(h)) # minibatch_size, n_labels * n_latent
        gamma_reshape = gamma.reshape((-1, self.n_latent)) # minibatch_size * n_labels, n_latent
        enum_label = torch.arange(0, self.n_labels).repeat((M)).view((-1, 1)) # minibatch_size * n_labels, 1
        h = self.decoder(gamma_reshape, enum_label.cuda())
        px_rate = self.px_decoder(h).reshape((M, self.n_labels, -1)) # (minibatch, n_labels, n_genes) 

        # add the dummy cell type
        eps = torch.nn.functional.softplus(self.eta).cuda() # n_genes <- this is the dummy cell type
        eps = eps.repeat((M, 1)).view(M, 1, -1) # (M, 1, n_genes)
        
        # account for gene specific bias and add noise
        r_hat = torch.cat([beta.unsqueeze(0).unsqueeze(1) * px_rate, eps], dim=1) # M, n_labels + 1, n_genes
        # now combine them for convolution
        px_rate = torch.sum(v.unsqueeze(2) * r_hat, dim=1) # batch_size, n_genes 

        px_o = self.px_o.cuda()

        return dict(
            px_o=px_o,
            px_rate=px_rate,
            eta=self.eta,
            gamma=gamma,
            v = v,
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
        y
            tensor of cell types (ignored)
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
        px_o = outputs["px_o"]
        eta = outputs["eta"]
        gamma = outputs["gamma"]
        v = outputs["v"]

        # reconstruction loss
        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_o)

        # global prior likelihood
        mean = torch.zeros_like(eta)
        scale = torch.ones_like(eta)

        glo_neg_log_likelihood_prior = -Normal(mean, scale).log_prob(eta).sum()        

       # gamma
        mean = torch.zeros_like(gamma)
        scale = 0.1 * torch.ones_like(gamma)

        neg_log_likelihood_prior = -Normal(mean, scale).log_prob(gamma).sum(1)

        # proportions
        # probs = v / v.sum(dim=1, keep_dims=True)
        # local_kl = kl(
        #     Categorical(probs=probs),
        #     Categorical(probs=self.y_prior.repeat(probs.size(0), 1)),
        # )
        local_kl = 0.0
        return reconst_loss + neg_log_likelihood_prior + local_kl, 0.0, glo_neg_log_likelihood_prior

