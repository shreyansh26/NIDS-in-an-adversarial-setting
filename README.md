# Network Intrusion Detection in an Adversarial Setting

This work is for my B.Tech thesis on the above topic. Broad breakdown of the work that is to be done -

## Semester - 1
* Understanding Adversarial Machine Learning
* Understanding and using the Cleverhans Python module
* Trying to fool Machine Learning based classifiers for NIDS by making them classify malicious network traffic as benign.


## Semester - 2
Yet to be decided, but should mainly focus on dealing with actaul raw network traffic and/or adversarial training.

-------------------------

## Requirements
* Python 3.x +
* [Cleverhans](https://github.com/tensorflow/cleverhans)
* Keras
* Tensorflow
* numpy
* pandas
* scikit_learn
* matplotlib

The exact versions of the modules can be found in the `requirements.txt` file.

## Steps to run
Since the code is still in the developmental phase, the current best version of the code is in the `test/` folder. To run enter the following commands on the terminal

```
$ sudo pip3 install -r requirements.txt
$ cd test
$ python3 for_graph.py

# If you want to see the training accuracy with epochs, run
$ python3 draw_graph.py
```
