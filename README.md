# MICC: Efficient Multi-Agent Communication via Mutual Information and Causal Consistency

This repository is the implementation of scmcc, we will open all results under our paper accepted!

We conduct the experiments using version SC2.4.6.2.69232, which is same as the SMAC run data release (https://github.com/oxwhirl/smac/releases/tag/v1).


## Installation
Set up StarCraft II and SMAC with the following command:

```bash
bash install_sc2.sh
```
It will download SC2.4.6.2.69232 into the 3rdparty folder and copy the maps necessary to run over. You also need to set the global environment variable:

```bash
export SC2PATH=[Your SC2 Path/StarCraftII]
```

Install Python environment with command:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install numpy scipy pyyaml pygame pytest probscale imageio snakeviz 
```