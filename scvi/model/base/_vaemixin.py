import logging
import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData
from torch.distributions import Normal

from ._log_likelihood import compute_elbo, compute_reconstruction_error

logger = logging.getLogger(__name__)


class VAEMixin:
    """Univseral VAE methods."""

    @torch.no_grad()
    def get_elbo(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> float:
        """
        Return the ELBO for the data.

        The ELBO is a lower bound on the log likelihood of the data used for optimization
        of VAEs. Note, this is not the negative ELBO, higher is better.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        elbo = compute_elbo(self.module, scdl)
        return -elbo

    @torch.no_grad()
    def get_marginal_ll(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_mc_samples: int = 1000,
        batch_size: Optional[int] = None,
    ) -> float:
        """
        Return the marginal LL for the data.

        The computation here is a biased estimator of the marginal log likelihood of the data.
        Note, this is not the negative log likelihood, higher is better.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_mc_samples
            Number of Monte Carlo samples to use for marginal LL estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        if hasattr(self.module, "marginal_ll"):
            log_lkl = 0
            for tensors in scdl:
                log_lkl += self.module.marginal_ll(tensors, n_mc_samples=n_mc_samples)
        else:
            raise NotImplementedError(
                "marginal_ll is not implemented for current model. "
                "Please raise an issue on github if you need it."
            )
        n_samples = len(indices)
        return log_lkl / n_samples

    @torch.no_grad()
    def get_reconstruction_error(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> Union[float, Dict[str, float]]:
        r"""
        Return the reconstruction error for the data.

        This is typically written as :math:`p(x \mid z)`, the likelihood term given one posterior sample.
        Note, this is not the negative likelihood, higher is better.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        reconstruction_error = compute_reconstruction_error(self.module, scdl)
        return reconstruction_error

    @torch.no_grad()
    def get_mmd_loss(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> float:
        """
        Computes the total MMD loss for the data over the given indices sliced by
        the given batch size.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        Float value containing the total MMD loss for the data
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        total_mmd = 0.0
        for tensors in scdl:
            _, _, scvi_loss = self.module(tensors)
            if hasattr(scvi_loss, "mmd"):
                total_mmd += scvi_loss.mmd.item()

        n_samples = len(scdl.indices)
        return total_mmd / n_samples

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        This is denoted as :math:`z_n` in our manuscripts.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.is_trained_ is False:
            warnings.warn("Model hasn't been trained yet, results will be random.")
        elif self.device.type != self.train_device_.type:
            warnings.warn(
                f"""Model trained with {self.train_device_} but currently on {self.device}. You may experience
                 reproducibility issues: https://github.com/YosefLab/scvi-tools/issues/1048.
                 This is expected behavior: https://github.com/pytorch/pytorch/issues/38219.
                 Move the model back to the train device for reproducibility
                 model.to_device('{self.train_device_}')"""
            )

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            z = outputs["z"]

            if give_mean:
                # does each model need to have this latent distribution param?
                if self.module.latent_distribution == "ln":
                    samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                    z = torch.nn.functional.softmax(samples, dim=-1)
                    z = z.mean(dim=0)
                else:
                    z = qz_m

            latent += [z.cpu()]
        return torch.cat(latent).numpy()
