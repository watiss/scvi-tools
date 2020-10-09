import torch
from torch.distributions import Normal, Categorical, kl_divergence as kl
import torch.nn as nn
import torch.nn.functional as F

from .classifier import Classifier
from ._base import Encoder, DecoderSCVI, FCLayers
from .utils import broadcast_labels
from .vae import VAE
from typing import Tuple, Dict

from scvi.core._log_likelihood import log_nb_positive_altparam


class VAEC(VAE):
    r"""
    A semi-supervised Variational auto-encoder model - inspired from M2 model.

    Described in (https://arxiv.org/pdf/1406.5298.pdf)

    Parameters
    ----------
    n_input :
        Number of input genes
    n_batch :
        Number of batches
    n_labels :
        Number of labels
    n_hidden :
        Number of nodes per hidden layer
    n_latent :
        Dimensionality of the latent space
    n_layers :
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate :
        Dropout rate for neural networks
    dispersion :
        One of the following

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational :
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood :
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    y_prior :
        If None, initialized to uniform probability over cell types

    Examples
    --------
    >>> gene_dataset = CortexDataset()
    >>> vaec = VAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
    ... n_labels=gene_dataset.n_labels)

    >>> gene_dataset = SyntheticDataset(n_labels=3)
    >>> vaec = VAEC(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
    ... n_labels=3)

    """

    def __init__(
        self,
        n_input,
        n_batch,
        n_labels,
        n_hidden=128,
        n_latent=10,
        n_layers=1,
        dropout_rate=0.1,
        y_prior=None,
        dispersion="gene",
        log_variational=True,
        gene_likelihood="nb",
    ):
        super().__init__(
            n_input,
            n_batch,
            n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
        )

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_cat_list=[n_labels],
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
        )

        self.decoder = FCLayers(
            n_in=n_latent,
            n_out=n_hidden,
            n_cat_list=[n_batch, n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )
        self.px_decoder = nn.Sequential(
        nn.Linear(n_hidden, n_input),
        nn.Softplus()
        )

    def generate_rate(self, z, batch_index=None, y=None):
        h = self.decoder(z, batch_index, y)
        return self.px_decoder(h)

    def inference(
        self, x, batch_index=None, y=None, n_samples=1, transform_batch=None
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        # getting library
        library = torch.sum(x, dim=1, keepdim=True)

        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)

        if transform_batch is not None:
            dec_batch_index = transform_batch * torch.ones_like(batch_index)
        else:
            dec_batch_index = batch_index

        h = self.decoder(z, batch_index, y)
        px_scale = self.px_decoder(h)
        px_rate = library * px_scale

        return dict(
            px_scale=px_scale,
            px_r=self.px_r,
            px_rate=px_rate,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            library=library,
        )

    def forward(self, x, local_l_mean, local_l_var, batch_index=None, y=None):
        # Sampling
        outputs = self.inference(x, batch_index, y)
        px_r = outputs["px_r"]
        px_rate = outputs["px_rate"]
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        reconst_loss = -log_nb_positive_altparam(x, px_rate, px_r).sum(dim=-1)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )
        
        return reconst_loss, kl_divergence_z, 0.0

