# multidms


**Running on ermine**

This notebook can be quite computationally expensive once you get to the simulation bits. It is reccomended that you run the fits on a high powered node with GPU enabled capabilities. To do this on the matsen group node `ermine`:

log into ermine, catching the port that you will run the notebook on
```
ssh -L 8080:localhost:8080 jgallowa@ermine.fhcrc.org
```
clone the repository
```
git clone git@github.com:matsengrp/multidms.git
```
create environment (reccomended) and install requirements
```
conda create --name multidms -y && conda activate multidms && conda install pip -y
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd multidms && pip install -r requirements.txt
```

Load the CUDA Modules:
```
module load CUDA/11.4.1
module load cuDNN/8.2.2.26-CUDA-11.4.1
```

run the notebook
```
jupyter notebook --no-browser --port=8080
```
Then, copy the localhost url (with the token) and paste it in your local browser.