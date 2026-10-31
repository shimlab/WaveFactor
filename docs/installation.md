# Installation Guide

## Standard Installation (Recommended)

Install WaveFactor with pip:

```bash
pip install wavefactor
```

Verify your installation:

```bash
python -c "import wavefactor as wf; print('WaveFactor version:', wf.__version__)"
```

!!! tip "Ready-to-Use Packages (Pre-Compiled)"
    Ready-to-use packages are provided for standard modern Python installations (**Python 3.10, 3.11, 3.12, and 3.13**) across common platforms (**Linux**, **Windows**, and **Mac**). 

    If these packages do not work for you, you can also try building from source (see below).

---

## Optional: Building from Source

If a pre-compiled wheel is not available or fails for your environment or you wish to compile locally:

### Prerequisites

These are illustrative reference settings rather than rigid constraints—other configurations may also work, though meeting them does not guarantee successful builds across arbitrary systems:

- **Python** $\ge 3.10$
- **CMake** $\ge 3.20$
- **A C++17 compiler** (e.g. GCC, Clang, or MSVC)

### Instructions

Clone the repository and install using `pip`:

```bash
# 1. Clone this repository
git clone https://github.com/shimlab/WaveFactor.git
cd WaveFactor

# 2. Build and install locally
pip install .
```
