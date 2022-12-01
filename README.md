# TuPaTE
Code for EMNLP 2022 paper "Efficiently Tuned Parameters are Task Embeddings"

### Setup
We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment by:

```shell
conda create -n tupate python=3.8.5
conda activate tupate
```

After we setup basic conda environment, install pytorch related packages via:

```shell
conda install -n pt2 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

Finally, install other python packages we need:

```shell
pip install -r requirements.txt
```

### Training
Run training scripts in [run_script](run_script) (e.g., RoBERTa for RTE):

```shell
bash run_script/run_rte_bert.sh
```

### Extract Task Embedding

Functions for extracting task embeddings for different parameter efficient tuning methods are provided in
```shell
extract_task_emb.py
```