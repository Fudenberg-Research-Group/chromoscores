# Chromoscores

A Python package created by Fudenberg research group for quantitative analysis of simulated Hi-C maps, providing tools to capture, process and evaluate chromatin interaction patterns such as Topoligically Associating Domains (TADs), flames, and peaks.


### Requirement 📃

### Installation 📦
First, 

```
git https://github.com/Fudenberg-Research-Group/chromoscores.git
```
```bash
pip install chromoscores
```

## Usage

```py
from chromoscores import BaseClass
from chromoscores import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m chromoscores
#or
$ chromoscores

```
### Analysis 📊
Observable features can be quantified, including:

- TADs (Topologically Associating Domains)
- flames
- Dots (loops between barriers)

  
See tutorials in `./jupyter_notebooks`.



[![codecov](https://codecov.io/gh/Fudenberg-Research-Group/chromoscores/branch/main/graph/badge.svg?token=chromoscores_token_here)](https://codecov.io/gh/Fudenberg-Research-Group/chromoscores)
[![CI](https://github.com/Fudenberg-Research-Group/chromoscores/actions/workflows/main.yml/badge.svg)](https://github.com/Fudenberg-Research-Group/chromoscores/actions/workflows/main.yml)



## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
