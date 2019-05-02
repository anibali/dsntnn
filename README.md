# PyTorch DSNT

This repository contains the official implementation of the differentiable
spatial to numerical (DSNT) layer and related operations.

```bash
$ pip install dsntnn
```

## Usage

Please refer to the [basic usage guide](examples/basic_usage.md).

## Scripts

### Running examples

```bash
$ python3 setup.py examples
```

HTML reports will be saved in the `examples/` directory. Please note that the `dsntnn` package must
be installed with `pip install` for the examples to run correctly.

### Building documentation

```bash
$ mkdocs build
```

### Running tests

Note: The dsntnn package must be installed before running tests.

```bash
$ pytest                                 # Run tests.
$ pytest --cov=dsntnn --cov-report=html  # Run tests and generate a code coverage report.
```

## Other implementations

* Tensorflow: [ashwhall/dsnt](https://github.com/ashwhall/dsnt)
  * Be aware that this particular implementation represents coordinates in the (0, 1)
    range, as opposed to the (-1, 1) range used here and in the paper.

If you write your own implementation of DSNT, please let me know so that I can add it to
the list. I would also *greatly* appreciate it if you could add the following notice
to your implementation's README:

> Code in this project implements ideas presented in the research paper
> "Numerical Coordinate Regression with Convolutional Neural Networks" by Nibali et al.
> If you use it in your own research project, please be sure to cite the
> original paper appropriately.

## License and citation

(C) 2017 Aiden Nibali

This project is open source under the terms of the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).

If you use any part of this work in a research project, please cite the following paper:

```bibtex
@article{nibali2018numerical,
  title={Numerical Coordinate Regression with Convolutional Neural Networks},
  author={Nibali, Aiden and He, Zhen and Morgan, Stuart and Prendergast, Luke},
  journal={arXiv preprint arXiv:1801.07372},
  year={2018}
}
```
