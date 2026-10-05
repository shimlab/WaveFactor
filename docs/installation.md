# Installation Guide

> **Note**: This documentation is a **work in progress** and subject to further improvements.

---

## 1. Requirements

Before installing, make sure you have:
- **Python** $\ge 3.7$
- **CMake** $\ge 3.16$
- **A C++17 compiler** (e.g. GCC or Clang)

---

## 2. Install in an Isolated Environment

Using an isolated virtual environment prevents conflicts with other Python packages:

```bash
# 1. Clone this repository (copy URL from the green "Code" button on GitHub)
git clone https://github.com/shimlab/WaveFactor.git
cd WaveFactor

# 2. Create and activate a virtual environment
python3 -m venv wavefactor-venv
source wavefactor-venv/bin/activate

# 3. Install WaveFactor
pip install .
```

---

## 3. (Optional) Check Installation Succeeded

Check that the installation succeeded:

```bash
python -c "import wavefactor; print('WaveFactor installed successfully')"
```

Run all tests (these should all pass if installation succeeded):

```bash
python tests/run_all_tests.py
```

---

## 4. (Optional) Deactivate or Remove

Deactivate the virtual environment by:

```bash
deactivate
```

To completely delete the installation, just remove the `wavefactor-venv/` folder:

```bash
rm -rf wavefactor-venv
```
