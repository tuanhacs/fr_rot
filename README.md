# Tree-Sliced Robust Optimal Transport

## Requirements

All tasks require the local `power_spherical` package. However, the dependency setup differs between the DDGAN task and the other tasks.

### DDGAN task

For the **DDGAN task**, create and activate the Conda environment from `environment.yaml`:

```bash
conda env create --file environment.yaml
conda activate twd
```

### Other tasks

For **all other tasks**, install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Install `power_spherical`

After setting up the corresponding environment, install the local `power_spherical` package:

```bash
cd power_spherical
pip install .
cd ..
```