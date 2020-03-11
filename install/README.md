Set up a conda environment:
```
conda create --name ice1d python=3.6
source activate ice1d
conda install -c conda-forge --file requirements.txt
```

add repository path to PYTHONPATH environement variable
``` export PYTHONPATH=$PYTHONPATH:[path to ice-model-1d] ```

Launch a notebook
```
jupyter-notebook code/lagrangian_dyn_model_1d.ipynb &
```
