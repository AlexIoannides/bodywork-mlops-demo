# Simulating the Lifecycle of a ML Pipeline on Kubernetes

![bodywork](https://bodywork-media.s3.eu-west-2.amazonaws.com/ml_system_testing.png)

This repository contains a Bodywork machine learning project that simulates the lifecycle of a train-and-deploy pipeline responding to new data undergoing concept drift. Each day a new tranche of synthetic data is simulated and used to test a model deployed as a model-scoring service. The new data is then combined with historical data and used to train a new model that will be used for the following day.
