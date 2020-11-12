import scvi

import numpy as np
from scvi.model import CondSCVI
from scvi.model.stereoscope import stVI
from scvi.data import register_tensor_from_anndata

# from scvi.data import register_tensor_from_anndata

dataset = scvi.data.pbmc_dataset(
            save_path="tests/data/10X",
            remove_extracted_data=True,
            run_setup_anndata=False,
        )

del dataset.obs["batch"]
scvi.data.setup_anndata(dataset, labels_key="str_labels")

n_latent = 2
n_hidden = 128
n_layers = 2
sparse=True

# train the conditional VAE
model = CondSCVI(dataset, n_latent=n_latent, n_layers=n_layers, n_hidden=n_hidden, sparse=sparse)
model.train(n_epochs=10, frequency=1)

# print(model.model.decoder.state_dict()["fc_layers.Layer 0.0.bias"])
# create some Gaussian noise and inject into the model for sampling
z = np.random.random_sample((100, n_latent)).astype(np.float32)
labels = np.zeros((100,1), np.long)

model.generate_from_latent(z, labels)

# get dataset ready
dataset.obs["indices"] = np.arange(dataset.n_obs)
register_tensor_from_anndata(dataset, "ind_x", "obs", "indices")

# now try to run the deconv algorithm on the spatial data
state_dict = (model.model.decoder.state_dict(), model.model.px_decoder.state_dict(), model.model.px_r.detach().cpu().numpy())
# add here number of cell type
spatial_model = stVI(dataset, dataset.uns["_scvi"]["summary_stats"]["n_labels"], state_dict, cell_type_prior=None,
                    n_latent=n_latent, n_hidden=n_hidden, n_layers=n_layers)
spatial_model.train(n_epochs=10, frequency=1)
# print(spatial_model.model.decoder.state_dict()["fc_layers.Layer 0.0.bias"])
# print(spatial_model.history)
spatial_model.get_proportions(dataset)