import scvi

import numpy as np
from scvi.model import CondSCVI
# from scvi.data import register_tensor_from_anndata

dataset = scvi.data.pbmc_dataset(
            save_path="tests/data/10X",
            remove_extracted_data=True,
            run_setup_anndata=True,
        )

# train the conditional VAE
model = CondSCVI(dataset)
model.train()

# create some Gaussian noise and inject into the model for sampling

z = np.random.random_sample((100, 10)).astype(np.float32)
labels = np.zeros((100,1), np.int32)
batches = np.zeros((100,1), np.int32)

model.generate_from_latent(z, batches, labels)
