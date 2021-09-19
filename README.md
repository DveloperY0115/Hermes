# Hermes

Hermes is an open source project aiming at building fully-autonomous driving system.

## Get Started

### 1. Create Python virtual environment & Install dependencies

Hermes is built on top of powerful frameworks and libraries such as:

- [Pytorch](https://pytorch.org)
- [Pytorch Lightning](https://www.pytorchlightning.ai)
- [pycls](https://github.com/facebookresearch/pycls)

The easiest way to prepare an environment which is ready to launch Hermes is to use [virtualenv](https://virtualenv.pypa.io/en/latest/). Note that this is not mandatory, but just a recommendation. You may still use other tools for managing Python dependencies such as [anaconda](https://www.anaconda.com), miniconda, of your preference. Instead, you would need to work a little bit (wouldn't be too tough!) since the codes were tested using `venv`.

To get started, open the shell, move working directory to the project root (where this file is located) and run:

```bash
virtualenv venv -p=3.8
source venv/bin/activate
pip install -r requirements.txt
```
