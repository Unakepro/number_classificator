#ifndef REGRESSION
#define REGRESSION

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <Eigen/Dense>



class log_regression {
    Eigen::MatrixXd x_train;
    Eigen::MatrixXd y_train;
    Eigen::MatrixXd x_test;
    std::vector<int8_t> y_test;
    
    Eigen::MatrixXd model;

    bool model_loaded = false;

public:
    log_regression(size_t model_r, size_t model_c);
    void train(size_t steps, double lr);

    int32_t get_prediction(Eigen::MatrixXd& img);
    void get_accuracy(const Eigen::MatrixXd& X, const std::vector<int8_t>& Y);

    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

    void compile(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, double mu, double sigma);

private:
    void random_weights(double mu, double sigma);

    Eigen::MatrixXd log_reg(Eigen::MatrixXd& X);
    double log_loss();
};

#endif