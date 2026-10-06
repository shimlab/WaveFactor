# WaveFactor: Bayesian Multiresolution Wavelet Spatial Factor Model

**WaveFactor** (formerly `WaviFM`) is a Bayesian factor modeling framework for spatial transcriptomics that explicitly models spatial length scales by performing Coordinate Ascent Variational Inference (CAVI) directly on 2D Discrete Wavelet Transform (DWT) coefficients.

---

## 🚀 Documentation

Please refer to the [documentation](https://shimlab.github.io/WaveFactor/) for installation, code snippets, examples, etc.

---

## 🧪 Testing & Quality Assurance

Run both the compiled C++ and Python test suite with the command:

```bash
# Run ALL tests (automatically compiles/checks C++ targets)
python tests/run_all_tests.py
```

Options:
- `python tests/run_all_tests.py --verbose` : Verbose test output.
- `python tests/run_all_tests.py -v`        : Verbose test output (same as `--verbose` flag).

---

## 📂 Repository Structure

- `docs/`: Documentation files
- `dev/`: Developer files
- `examples/`: Example scripts
- `lib`: External libraries
- `src/`: C++ CAVI engine source files
- `test/`: C++ testing suite
- `tests/`: Python test suite
- `wavefactor/`: Python package