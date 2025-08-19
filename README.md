# C++ Artificial Intelligence

This project is an exercise in implementing various machine-learning and deep-learning algorithms from scratch in C++. The goal is to understand the internals of these algorithms, how they work, and how to optimize them, while honing skills in C++.

## Build and run

From the repo root:

```bash
# From repo root
./build.sh            # Release, ML-only (default)
./build.sh Debug      # Debug, ML-only
./build.sh --with-dl  # Release, include WIP DL sources (may not compile yet)
```

The script configures with CMake, builds, and then runs the `cpp_ai` executable. It also honors `VCPKG_ROOT` if set to override the vcpkg toolchain path.

## Machine Learning Algorithms

1. Linear Regression: Implemented. A simple yet powerful algorithm used for predicting a continuous target variable.
2. Decision Trees: Not Implemented. An intuitive algorithm that makes decisions based on certain conditions.
3. k-Nearest Neighbors: Not Implemented. An algorithm that classifies a data point based on how its neighbors are classified.
4. Support Vector Machines (SVM): Not Implemented. A powerful classification algorithm that finds the optimal hyperplane that separates different classes.
5. K-Means Clustering: Not Implemented. An unsupervised algorithm that groups similar data points together.

## Deep Learning Algorithms

1. Neural Networks: Not Implemented. A foundational algorithm in deep learning that tries to mimic the workings of a human brain to make decisions.
2. Convolutional Neural Networks (CNN): Not Implemented. A type of neural network particularly effective in tasks related to image processing.
3. Recurrent Neural Networks (RNN): Not Implemented. A type of neural network with 'memory' for use cases where the order of data is important.
4. Autoencoders: Not Implemented. A type of artificial neural network used for learning efficient codings of input data.
5. Generative Adversarial Networks (GANs): Not Implemented. An algorithmic structure where two neural networks contest with each other in a zero-sum game framework.

## Usage

Each algorithm will be implemented as a C++ class. This means you can create an instance of the algorithm, fit it to your data using the `train` method, and make predictions with the `predict` method. For example, see LinearRegression.cpp and main.cpp.

## Contribution

Feel free to contribute to this project by implementing the not implemented algorithms or by optimizing the current implementations. Open a pull request and let's discuss your changes.

## License

This project is licensed under the terms of the MIT license.
