# Kilter Board Grade Prediction

# How to Run

Clone the repository with

`git clone https://github.com/rhit-fernanms/Kilter-Board-Grade-Prediction.git`

and navigate to the corresponding directory.

## Dependencies
Install the `uv` python environment manager with
```bash
cd src
curl -LsSf https://astral.sh/uv/install.sh | sh 
source $HOME/.local/bin/env
```

You may need to close and reopen your terminal for your path to be updated. Now within the `src` directory, run

```bash
uv init --python 3.12 --no-package
uv venv
source .venv/bin/activate
```

to create a python environment. To grab all required dependencies, run

```bash
uv pip install torch
uv pip install -r requirements.txt
```

## Generate data

From the base directory, create a folder called `data/images` and run the image generation script with

```bash
source src/.venv/bin/activate
python climbs_to_pngs.py
```

This will save around 36,000 climb images (5.2Gb) to the `data/images` directory. If you wish to cut down on the number of images, change the `NUM_IMAGES` constant to the number of images you desire. 

## Train Model

Run the Jupyter notebooks `ResNetTransfer.ipynb` or `ResNetTransfer_regressor.ipynb` to train the classification or regression models respectively. Currently, both models train on only 12,000 of the 36,000 images due to limited memory, but if you wish to change this number, navigate to the `collect_data` function within `collect_data.py` and set the loop to break at the number you desire.