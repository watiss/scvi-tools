import scvi

import os
print(os.getcwd())

import numpy as np
from scvi.model.stereoscope import scStereoscope, stStereoscope
from scvi.data import register_tensor_from_anndata

dataset = scvi.data.pbmc_dataset(
            save_path="tests/data/10X",
            remove_extracted_data=True,
            run_setup_anndata=True,
        )

dataset.obs["indices"] = np.arange(dataset.n_obs)
register_tensor_from_anndata(dataset, "ind_x", "obs", "indices")

model = scStereoscope(dataset)
model.train(n_epochs=100, frequency=1)
params = model.get_params()


model = stStereoscope(dataset, params)
model.train(n_epochs=100, frequency=1)
print(model.get_proportions())