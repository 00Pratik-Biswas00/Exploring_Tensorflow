## Course link ->

https://developers.google.com/learn/pathways/tensorflow#codelab-https://developers.google.com/codelabs/tensorflow-2-computervision

## Answers

#### 1.

When using model.predict, you may get an answer close to, but not exactly what you expect. You can attribute that to neural networks generally dealing in -> Probabilities

#### 2.

Sequential defines a sequence of layers in the neural network.
Flatten takes a square and turns it into a one-dimensional vector.
Dense adds a layer of neurons.
Activation functions tell each layer of neurons what to do. There are lots of options, but use these for now:
Relu effectively means that if X is greater than 0 return X, else return 0. It only passes values of 0 or greater to the next layer in the network.
Softmax takes a set of values, and effectively picks the biggest one.

#### 3.

Pooling - reduces the overall amount of information in an image while maintaining the detected features as present.

#### 4.

Overfitting occurs as a result of a neural network trained with - limited data
