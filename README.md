# Number Classificator

**Number Classificator** is a simple digit recognition application created using SFML in C++. The app allows users to draw digits (0–9) in an input field, and it attempts to identify the number using a machine learning model trained on the MNIST dataset. This project serves as an example of applying logistic regression for image recognition within a C++.


## Features

- **Digit Drawing Interface**: Users can draw a digit between 0 and 9.
- **Real-Time Recognition**: The app uses logistic regression to classify the drawn number.

## Limitations

This recognition system is based on logistic regression trained on the MNIST dataset. On the test dataset, it achieves an accuracy of around 91.5%, which is close to the limit for this method. While some additional preprocessing and extended training might improve performance slightly, logistic regression has inherent limitations for this task.

A multi-layer neural network would be a better choice for achieving higher accuracy, but this model was created just to showcase the use C++ in implementing basic machine learning concepts, obviously MNIST resolution is lower, so we have to compress our drawen image and logistic regression is definetly not the best choice, but I'm just trying to show that you can relativly simply implement lositic regression via c++ and do a pretty cool thing with it.

## Configuration
Model has 785 parameters - 784 for each pixes and the last one for bias.
Initial parameters was set random from 0 to 0.0001. 
Learning rate is 0.5 and around 8000 epoch was taken to train it.


## Prerequisites

Before you begin, you should install the (sfml library)[https://www.sfml-dev.org/] 

## Installation

1. **Clone the Repository**  
```bash
git clone https://github.com/yourusername/number_classificator.git
cd number_classificator
```
   
2. **Create Build Directory and Configure Project with CMake**
```bash
mkdir build
cmake -S .  -B build
```

3. **Make the project**
```bash
cd build
make
```

4. **Run it**
```bash
cd ..
./build/bin/ProjExec
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
