# WaveFactor: Bayesian Multiresolution Wavelet Spatial Factor Model

**WaveFactor** (formerly `WaviFM`) is a Bayesian factor modeling framework for spatial transcriptomics that explicitly models spatial length scales by performing Coordinate Ascent Variational Inference (CAVI) directly on 2D Discrete Wavelet Transform (DWT) coefficients.

---

## Quick Install

Install WaveFactor with pip:

```bash
pip install wavefactor
```

Verify your installation:

```bash
python -c "import wavefactor as wf; print('WaveFactor version:', wf.__version__)"
```

---

## Documentation

Please refer to the [documentation](https://shimlab.github.io/WaveFactor/) for installation, code snippets, examples, etc.

---

## Testing & Quality Assurance

Run both the compiled C++ and Python test suite with the command:

```bash
# Run ALL tests (automatically compiles/checks C++ targets)
python tests/run_all_tests.py
```

Options:
- `python tests/run_all_tests.py --verbose` : Verbose test output.
- `python tests/run_all_tests.py -v`        : Verbose test output (same as `--verbose` flag).

---

## Repository Structure

- `docs/`: Documentation files
- `dev/`: Developer files
- `examples/`: Example scripts
- `lib`: External libraries
- `src/`: C++ CAVI engine source files
- `test/`: C++ testing suite
- `tests/`: Python test suite
- `wavefactor/`: Python package

---

## Citations, Publications and Presentations

**Preprint**: Coming soon, stay tuned!

**Research Poster (2026 November)**: [View PDF](https://github.com/shimlab/WaveFactor/blob/main/docs/assets/2026Nov-WaveFactor-Poster.pdf)

---

## Contacts

**Authors**

- Yulin Wu¹²³
  - Email: [yulin4@student.unimelb.edu.au](mailto:yulin4@student.unimelb.edu.au)
  - LinkedIn: [www.linkedin.com/in/yulin-wu-59a0591b0](https://www.linkedin.com/in/yulin-wu-59a0591b0/)
- Heejung Shim¹²³⁴⁵⁶⁷

**Affiliations**

¹ Melbourne Integrative Genomics, The University of Melbourne  
² School of Mathematics and Statistics, The University of Melbourne  
³ ARC Centre of Excellence for the Mathematical Analysis of Cellular Systems, The University of Melbourne  
⁴ Snow Vision Accelerator, The University of Sydney  
⁵ Charles Perkins Centre, The University of Sydney  
⁶ Sydney Precision Data Science Centre, The University of Sydney  
⁷ School of Mathematics and Statistics, The University of Sydney
