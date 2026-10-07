# ProtoPlanetary Disk AutoEncoders

An AutoEncoder model to reconstruct and generate new images of edge-on Proto Planetary
disks using physical parameters as input ([Telkamp et al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac96f1))

The model architecture is the following:

![AE model](https://github.com/jorgemarpa/PPDAE/blob/main/figures/PPDAE_arch_V2.png)

### Image Samples

The edge-on disk images used for training were generated using the MCFOST Radiative
Transfer code. The training set looks like this:

![imgs](https://github.com/jorgemarpa/PPDAE/blob/main/figures/image_wall.png)

with a wide variety of shapes and sizes.

### Physical Parameters

Each image is associated with the physical parameters used to simulate the source.
We used 8 physical parameters to as inputs to MCFOST:

```python
    m_dust = 'mass of the dust'
    Rc     = 'critical radius when exp drops(size)'
    f_exp  = 'flare exponent'
    H0     = 'scale hight'
    Rin    = 'inner raidus'
    sd_exp = 'surface density exponent'
    alpha  = 'dust stettling'
    inc    = 'inclination'
```

All physical parameters where sampled from a evenly spaced grid and a random sampling
between the range of possible values in order to fill up the gaps.

![phy dist](https://github.com/jorgemarpa/PPDAE/blob/main/figures/phy_params.png)

## Usage

Use `ae_main.py` to train a AE model with the following parameters:
```python
    -h, --help            show this help message and exit
    --dry-run             Load data and initialize models [False]
    --machine MACHINE     were to is running (local, [colab], exalearn)
    --data DATA           data used for training (MNIST, [PPD])
    --img-norm IMG_NORM   type of normalization for images (global, [image])
    --par-norm PAR_NORM   physical parameters are 0-1 scaled ([T],F)
    --subset SUBSET       data subset ([25052021], fexp1)
    --optim OPTIM         Optimizer ([Adam], SGD)
    --lr LR               learning rate [1e-4]
    --lr-sch LR_SCH       learning rate shceduler ([None], step, exp, cosine, plateau)
    --batch-size BATCH_SIZE
                          batch size [128]
    --num-epochs NUM_EPOCHS
                          total number of training epochs [100]
    --early-stop          Early stoping
    --cond COND           physics conditioned AE (F,[T])
    --feed-phy FEED_PHY   feed physics to decoder ([F],T)
    --latent-dim LATENT_DIM
                          dimension of latent space [8]
    --dropout DROPOUT     dropout for all layers [0.2]
    --kernel-size KERNEL_SIZE
                          2D conv kernel size, encoder [3]
    --conv-blocks CONV_BLOCKS
                          conv+actfx+pool blocks [5]
    --model-name MODEL_NAME
                          name of model [ConvLinTrans_AE]
    --comment COMMENT     extra comments
```

### Reconstruction examples

Edge-on images (upper row), AE reconstruction (middle), and residuals (lower row).

![recon](https://github.com/jorgemarpa/PPDAE/blob/main/figures/Test_Recon_106050_52a73755.png)

Training logs and models are stored at W&B here:
 https://wandb.ai/deep_ppd/PPD-AE

## Dependencies

We use `poetry` to manage dependencies and environment. First install poetry:

```
pip install poetry
```

Then install the necessary libraries listed in `poetry.toml` using the install command from
poetry inside the repo directory:
```
poetry install
```

More info on how to use `poetry` -> https://python-poetry.org/docs/

## Sources and inspiration

* https://www.jeremyjordan.me/variational-autoencoders/
* https://github.com/wiseodd/generative-models
* https://github.com/jorgemarpa/PELS-VAE
