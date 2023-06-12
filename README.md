# Deployment of ML models using FastAPI and Docker



## About this Repo
This repository contains miscellaneous machine learning scripts and notebooks for data analysis, modeling, and visualization. 





# Dataset
## Breast Cancer Wisconsin (Diagnostic) Data Set

For this workshop we are going to work with the following dataset:

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

### Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

## Preparing the environment

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/iamirmasoud/fastapi_ml_docker
cd fastapi_ml_docker
```

2. Create (and activate) a new environment, named `fastapi_ml` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	```shell
	conda create -n fastapi_ml python=3.10
	source activate fastapi_ml
	```
	
	At this point your command line should look something like: `(fastapi_ml) <User>:fastapi_ml_docker <user>$`. The `(fastapi_ml)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You can install dependencies using:
```
pip install -r requirements.txt
```


# Train

After we installing all the dependencies we can now run the script in code/train.py, this script takes the input data and outputs a trained model and a pipeline for our web service.

`$ python code/train.py`

# Web application

Finally we can test our web application by running:

`$ uvicorn main:app`

# Docker

Now that we have our web application running, we can use the Dockerfile to create an image for running our web application inside a container

`$ docker build . -t sklearn_fastapi_docker`

And now we can test our application using Docker

`$ docker run -p 8000:8000 sklearn_fastapi_docker`

# Test!

Test by using the calls in tests/example_calls.txt from the terminal