Set up a conda environment:
```
conda create --name ice1d python=3.6
source activate ice1d
conda install -c conda-forge --file requirements.txt
```

Launch the notebook
```
jupyter-notebook ../code/lagrangian_dyn_model_1d.ipynb &
```
