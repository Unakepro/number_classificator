# Number Classificator

**Number Classificator** is a simple digit recognition application created using SFML in C++. The app allows users to draw digits (0–9) in an input field, and it attempts to identify the number using a machine learning model trained on the MNIST dataset. This project serves as an example of applying logistic regression for image recognition within a C++.


## Features

- **Digit Drawing Interface**: Users can draw a digit between 0 and 9.
- **Real-Time Recognition**: The app uses logistic regression to classify the drawn number.

## Limitations

This recognition system is based on logistic regression trained on the MNIST dataset. On the test dataset, it achieves an accuracy of around 91.5%, which is close to the limit for this method. While some additional preprocessing and extended training might improve performance slightly, logistic regression has inherent limitations for this task.

A multi-layer neural network would be a better choice for achieving higher accuracy, but this model was created just to showcase the use C++ in implementing basic machine learning concepts, obviously MNIST resolution is lower, so we have to compress our drawen image and logistic regression is definetly not the best choice, but I'm just trying to show that you can relativly simply implement lositic regression via c++ and do a pretty cool thing with it.

## Configuration

The model consists of 785 parameters: 784 parameters correspond to each pixel in the input image, and last one is a bias.

- **Parameter Initialization**: The initial parameters were randomly set within the range of 0 to 0.0001.
- **Learning Rate**: The learning rate is set to 0.5.
- **Training Epochs**: The model was trained for approximately 8,000 epochs.


## Prerequisites

Before you begin, you should install the [SFML](https://www.sfml-dev.org/) graphic library.

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

## Demo



https://github.com/user-attachments/assets/3a6e44e7-60c9-4ea5-8b59-b181f9880f17




## License

[MIT](https://choosealicense.com/licenses/mit/)
