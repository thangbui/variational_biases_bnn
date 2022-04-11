This repo contains some sample code to run AIS and MFVI on simple Bayesian neural networks.

### Set up

The AIS implementation is largely based on Xuechen Li's [BDMC package](https://github.com/lxuechen/BDMC).
We first checkout this package in `third`:

```bash
cd third
git clone git@github.com:lxuechen/BDMC.git
```

This also requires `numpy`, `matplotlib`, `pytorch` and `tqdm`. These can be installed by using e.g.

```bash
pip install numpy torch tqdm matplotlib
```

### Example

We can now run AIS and MFVI on a toy regression task to create figure 2 in the [workshop paper](http://bayesiandeeplearning.org/2021/papers/62.pdf):

```bash
python example.py
```

The output should look like [this file](example.output) . The predictions are included under `tmp`.