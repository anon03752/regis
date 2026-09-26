# Pretrained checkpoints

- `regis_k3_seed1.pt`: Ising REGIS, training seed 1, step 30,000.
- `mnist_regis_seed3.pt`: cycling-MNIST REGIS, run index 3 (training seed 45), step 100,000.
- `hearts_regis_seed0.pt`: zebrafish-heart REGIS, training seed 0, step 100,000.

Each checkpoint contains EMA generator weights and model settings for inference.

`mnist_starts.npy` contains four zero images from the MNIST test split, taken from the project page's sample bank. Each image is resized to 22×22 and padded to 32×32, matching the training preprocessing.

`hearts_starts.npz` contains four wounded uninjured hearts: the cell-type, damage and alive channels of the model's starting state, and the section masks. They are sections of the regeneration atlas of Li et al. 2025, Nat. Commun. 16:3716 (Zhao et al., Zenodo 14991776, CC BY 4.0), rebuilt by `domains/hearts/data_cohort.py` and wounded at the apex as in training.
