# All You Need is a Good Functional Prior for Bayesian Deep Learning

Companioning code for the JMLR submission *"All You Need is a Good Functional Prior for Bayesian Deep Learning"*.


## Environment setup

For the first environment setup, you will need to create a Docker image with all requirements for this package. To do so, if not available on your machine, please install Docker following the [official installation guide](https://docker.com).
Once Docker is installed, you can proceed to build the image:

```bash
make docker image
```

If successful, you will end up with a Docker image called `optbnn:latest`.

### Test the environment

To check that everything works properly, you can run an interactive session on the Docker image just built and navigate in the container (all modifications will be undone on container exit). To do so, you can simply run 

```bash
make docker run-{python,interactive,jupyter}
```

## Running Demos

Firstly, you need to launch the jupyter lab by executing the following command

```bash
make run-jupyter
```

Once jupuyter lab is launched, demo notebooks can be found in the `notebooks` directory. Here, we included some demos as follows

- `1D_regression_Gaussian_prior.ipynb`: Comparison between `FG` and `GPiG` priors on a 1D regression data.
- `1D_regression_hierarchical_prior.ipynb`: Comparison between `FH` and `GPiH` priors on a 1D regression data set.
- `1D_regression_norm_flow_prior.ipynb`: Comparison between `Fixed NF` and `GPiNF` priors on 1D regression data.
- `2D_classification.ipynb`: The effect of using different configurations of the target GP prior to the predictive posterior on a 2D classification task.
- `2D_classification_hierarchical_gp_prior.ipynb`: The effect of using a target hierarchical-GP prior to the predictive posterior on a 2D classification task.
- `uci_regression.ipynb`: Comparison between `FG` and `GPiG` priors on a UCI data set.

