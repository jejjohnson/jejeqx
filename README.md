# My Personal JAX Library

[![codecov](https://codecov.io/gh/jejjohnson/eqx-trainer/branch/master/graph/badge.svg?token=DM1DRDASU2)](https://codecov.io/gh/jejjohnson/eqx-trainer)
[![CodeFactor](https://www.codefactor.io/repository/github/jejjohnson/eqx-trainer/badge)](https://www.codefactor.io/repository/github/jejjohnson/eqx-trainer)

This is my personal library where I have all of my code that uses jax and the equinox backend.

Pronounced: Jay-EE-JEX

---
## Components

**Versioning**: `versioneer`

**Documentation**: `jupyterbook`


---
## Installation

This package isn't pip-worthy (yet) but here are a few options for installation.

**Option I**: Use the `pip` install option (locally)

```bash
https://github.com/jejjohnson/jemanjjax.git
cd jemanjjax
pip install -e .[dev,all]
```

**Option II**: Install it from pip directly.

```bash
pip install "git+https://github.com/jejjohnson/jemanjjax.git"
```

---
## External Packages

I use quite a few of external packages that I've relegated to their own repo.

**Neural Fields**

```bash
pip install "git+https://github.com/jejjohnson/eqx-nerf.git"
```

**Trainer**

```bash
pip install "git+https://github.com/jejjohnson/eqx-trainer.git"
```


**OceanBench**

```bash
brew install g++ cmake eigen boost gsl
pip install "git+https://github.com/jejjohnson/oceanbench.git"
```



---
---
## Inspiration

* [UVADLC Course](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide4/Research_Projects_with_JAX.html)
* [JaxLightning](https://github.com/ludwigwinkler/JaxLightning)
* [Tez](https://github.com/abhishekkrthakur/tez)
