# Bayesian Optimization and Active Learning in Drug Discovery

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://hfooladi.github.io/bayesoptimol/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About the Project

**bayesoptimol** is an interactive educational resource that explores the application of Bayesian optimization and active learning techniques in molecular discovery. This project aims to bridge the gap between theoretical machine learning concepts and their practical applications in drug discovery through clear explanations and executable code examples.

## Features

- **Interactive Jupyter Book**: Comprehensive content with theoretical foundations and practical implementations
- **Google Colab Integration**: Run all code examples directly in your browser without setup
- **Hands-on Examples**: Learn by doing with real-world molecular datasets and scenarios
- **From Basics to Advanced**: Progressive learning path from fundamentals to cutting-edge methods

## Topics Covered

- Bayesian optimization fundamentals
- Gaussian processes for molecular property prediction
- Acquisition functions for molecular optimization
- Active learning strategies in drug discovery
- Molecular representations for machine learning
- Case studies with real-world applications
- Advanced topics (batch optimization, multi-objective optimization, etc.)

## Getting Started

### Read Online

Visit the hosted version of the book at: [https://hfooladi.github.io/bayesoptimol/](https://hfooladi.github.io/bayesoptimol/)

### Run Locally

To build and run the book locally:

```bash
# Clone the repository
git clone https://github.com/HFooladi/bayesoptimol.git
cd bayesoptimol

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build the book
cd docs
jupyter-book build .

# Open the HTML output
open _build/html/index.html  # On macOS
# or
start _build/html/index.html  # On Windows
# or
xdg-open _build/html/index.html  # On Linux
```

## Structure

```
bayesoptimol/
├── _config.yml          # Configuration for Jupyter Book
├── _toc.yml             # Table of contents structure
├── intro.md             # Introduction to the book
├── chapters/            # Book chapters
│   ├── foundations/     # Foundational concepts
│   ├── gaussian_processes/  # GP-related content
│   └── ...              # Other chapters
├── references.bib       # Bibliography
├── requirements.txt     # Python dependencies
└── README.md            # Project information
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for discussion.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citations

If you find this resource useful in your research, please consider citing:

```
@book{fooladi2025bayesoptimol,
  title={Bayesian Optimization and Active Learning in Drug Discovery},
  author={Fooladi, Hosein},
  year={2025},
  publisher={GitHub},
  url={https://github.com/HFooladi/bayesoptimol}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Jupyter Book](https://jupyterbook.org/) for the publishing framework
- [RDKit](https://www.rdkit.org/) for cheminformatics functionality
- [GPyTorch](https://gpytorch.ai/) and [BoTorch](https://botorch.org/) for Bayesian optimization implementations
- All contributors and the open-source community

## Contact

Hosein Fooladi - [GitHub Profile](https://github.com/HFooladi)

Project Link: [https://github.com/HFooladi/bayesoptimol](https://github.com/HFooladi/bayesoptimol)