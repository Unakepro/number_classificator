#include "mnist_regression.hpp"
#include <random>

std::random_device rd;
std::mt19937 gen(rd());


log_regression::log_regression(size_t model_r, size_t model_c): model_loaded(false) {
    model = Eigen::MatrixXd(model_r, model_c);
}

void log_regression::train(size_t steps, double lr) {
    auto Y_T = y_train.transpose();
    auto X_T = x_train.transpose();

    for(size_t i = 0; i < steps; ++i) {

        auto der = (1.0 / x_train.cols()) * ((log_reg(x_train) - Y_T) * X_T);
        model.noalias() -= der * lr;

        std::cout << "Loss " << log_loss() << std::endl;
    }
}

int32_t log_regression::get_prediction(Eigen::MatrixXd& img) {
    auto pred = log_reg(img);
    int row;
    pred.col(0).maxCoeff(&row);

    return row;
}

void log_regression::get_accuracy(const Eigen::MatrixXd& X, const std::vector<int8_t>& Y) {
    x_test = X;
    y_test = Y;

    auto pred = log_reg(x_test);
    
    Eigen::VectorXi maxIndices(pred.cols());
    for(size_t i = 0; i < pred.cols(); ++i) {
        int row;
        pred.col(i).maxCoeff(&row);

        maxIndices(i) = row;
    }

    double guessed = 0;

    if(y_test.size() != maxIndices.size()) {
        throw std::logic_error("wrong size!");
    }

    for(size_t i = 0; i < maxIndices.size(); ++i) {
        if(maxIndices[i] == static_cast<int>(y_test[i])) {
            guessed += 1;
        }
    }

    std::cout << "Accuracy: " << (guessed / x_test.cols()) * 100 << " %" << std::endl;
}

void log_regression::saveModel(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);

    if(file.is_open()) {
        size_t rows = model.rows();
        size_t cols = model.cols();
        

        file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<char*>(&cols), sizeof(cols));

        file.write(reinterpret_cast<char*>(model.data()), sizeof(double) * rows * cols);
        file.close();

        std::cout << "Model saved to " << filename << std::endl;
    }
    else {
        throw std::logic_error("Not possible to open file!");
    }
}

void log_regression::loadModel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open()) {
        size_t rows;
        size_t cols;


        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        file.read(reinterpret_cast<char*>(model.data()), sizeof(double) * rows * cols);
        file.close();

    }
    else {
        throw std::logic_error("Not possible to open file!");
    }

    model_loaded = true;
}

void log_regression::compile(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, double mu, double sigma) {
    x_train = X; 
    y_train = Y;
    
    if(!model_loaded) {
        random_weights(mu, sigma);
    }
}

void log_regression::random_weights(double mu, double sigma) {
    std::uniform_real_distribution<> dis(mu, sigma);

    for(size_t i = 0; i < model.rows(); ++i) {
        for(size_t j = 0; j < model.cols(); ++j) {
            model(i, j) = dis(gen);
        }
    }
}

Eigen::MatrixXd log_regression::log_reg(Eigen::MatrixXd& X) {
    return 1.0 / (1.0 + ((-(model * X).array()).exp()));
}

double log_regression::log_loss() {
    double loss = (-1.0 / x_train.cols()) * ((y_train.array() * (log_reg(x_train).transpose().array().log())) + 
    (1 - y_train.array()) * (1-log_reg(x_train).transpose().array()).log()).sum();
    
    return loss;
}