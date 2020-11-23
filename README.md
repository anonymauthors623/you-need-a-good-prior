# All you need is a good prior

Companioning code for the JMLR submission "All you need is a good prior".


## 0. Environment setup

For the first environment setup, you will need to create a Docker image with all requirements for this package. To do so, if not available on your machine, please install Docker following the [official installation guide](https://docker.com).
Once Docker is installed, you can proceed to build the image:

```bash
make docker image
```

If successful, you will end up with a Docker image called `optbnn:latest`.

### 0.1 Test the environment

To check that everything works properly, you can run an interactive session on the Docker image just built and navigate in the container (all modifications will be undone on container exit). To do so, you can simply run 

```bash
make docker run-{python,interactive,jupyter}
```




