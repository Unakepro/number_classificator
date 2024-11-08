#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include <random>
#include <SFML/Graphics.hpp>
#include "regression/mnist_regression.hpp"



int32_t bigEndianToLittle (int32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x000000FF) << 24);
}

std::vector<int8_t> reader_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open()) {
        int32_t magicNumber;
        int32_t num_items;

        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));

        magicNumber = bigEndianToLittle(magicNumber);
        num_items = bigEndianToLittle(num_items);

        std::cout << "Magic Number: " << magicNumber << " " << std::endl;
        std::cout << "Number of labels: " << num_items << " " << "\n\n\n";

        std::vector<int8_t> labels(num_items);
        file.read(reinterpret_cast<char*>(labels.data()), num_items);
        
        return labels;
    }
    else {
        throw std::logic_error("Not possible to open file!");
    }
}


void reader_images(const std::string& filename, Eigen::MatrixXd& matrix) {
    std::ifstream file(filename, std::ios::binary);

    if (file.is_open()) {
        int32_t magicNumber;
        int32_t num_items;
        int32_t rows;
        int32_t cols;

        file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        magicNumber = bigEndianToLittle(magicNumber);
        num_items = bigEndianToLittle(num_items);
        rows = bigEndianToLittle(rows);
        cols = bigEndianToLittle(cols);

        std::cout << "Magic Number: " << magicNumber << " " << std::endl;
        std::cout << "Number of images: " << num_items << " " << std::endl;
        std::cout << "Number of rows: " << rows << " " << std::endl;
        std::cout << "Number of cols: " << cols << " " << "\n\n\n";

        std::vector<uint8_t> raw_data(rows * cols * num_items);
        file.read(reinterpret_cast<char*>(raw_data.data()), raw_data.size());

        for(size_t i = 0; i < num_items; ++i) {
            for(size_t j = 0; j < (rows * cols); ++j) {
                if(static_cast<int>(raw_data.at(i*(rows*cols)+j)) < 0) {
                    throw std::logic_error("wrong");
                }
                matrix(j, i) = static_cast<double>(raw_data.at(i*(rows*cols)+j)) / 255.0;
            }
            matrix(rows * cols, i) = 1;
        }
    }
    else {
        throw std::logic_error("Not possible to open file!");
    }

}


void one_hot(Eigen::MatrixXd& matrix, const std::string& filename, size_t n_classes) {
    auto labels = reader_labels(filename);

    matrix = Eigen::MatrixXd::Zero(labels.size(), n_classes);

    for(size_t i = 0; i < matrix.rows(); ++i) {
        matrix.row(i).col(labels[i]) << 1;
    }
}


int main() {
    Eigen::MatrixXd x_train(785, 60000);
    Eigen::MatrixXd y_train(60000, 10);

    reader_images("mnist/train-images.idx3-ubyte", x_train);
    one_hot(y_train, "mnist/train-labels.idx1-ubyte", 10);

    log_regression lg(10, 785);

    lg.loadModel("mnist.model");
    //lg.compile(x_train, y_train, 0, 0.0001);
    //lg.train(10, 0.5);
    lg.saveModel("mnist.model");

    Eigen::MatrixXd x_test(785, 10000);

    reader_images("mnist/test-images.idx3-ubyte", x_test);
    auto y_test = reader_labels("mnist/test-labels.idx1-ubyte");

    lg.get_accuracy(x_test, y_test);
   
    sf::RenderWindow window(sf::VideoMode(500, 280), "28x28 Drawing Input");
    
    sf::RenderTexture drawingCanvas;
    drawingCanvas.create(280, 280);
    drawingCanvas.clear(sf::Color::Black);

    sf::Sprite brushField(drawingCanvas.getTexture());
    brushField.setPosition(0, 0);

    sf::CircleShape brush(8);
    brush.setFillColor(sf::Color::White);

    sf::RectangleShape clearButton(sf::Vector2f(220, 40));
    clearButton.setPosition(280, 240);
    clearButton.setFillColor(sf::Color(150, 150, 150));

    sf::Text clearButtonText;
    sf::Font font;

    
    if(!font.loadFromFile("fonts/RobotoCondensed.ttf")) {
        throw std::logic_error("Font not found!");
    }

    clearButtonText.setFont(font);
    clearButtonText.setString("Clear");
    clearButtonText.setCharacterSize(20);
    clearButtonText.setFillColor(sf::Color::White);
    clearButtonText.setPosition(370, 248);


    sf::RectangleShape numberField(sf::Vector2f(220, 240));
    numberField.setPosition(280, 0);
    numberField.setFillColor(sf::Color(220, 220, 220));

    sf::Text numberText;
    numberText.setFont(font);
    numberText.setCharacterSize(150);
    numberText.setFillColor(sf::Color::Black);
    numberText.setPosition(350, 20);
    numberText.setString("0");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
        }

        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            sf::Vector2i mousePos = sf::Mouse::getPosition(window);
            
            if(clearButton.getGlobalBounds().contains(mousePos.x, mousePos.y)) {
                drawingCanvas.clear();
                drawingCanvas.display();
            }
            else if(brushField.getGlobalBounds().contains(mousePos.x, mousePos.y)) {
                brush.setPosition(static_cast<float>(mousePos.x) - brush.getRadius(), static_cast<float>(mousePos.y) - brush.getRadius());
                drawingCanvas.draw(brush);  
                drawingCanvas.display();


                sf::Image capturedImg = drawingCanvas.getTexture().copyToImage();
                Eigen::MatrixXd imageMatrix(785, 1);

                for (size_t y = 0; y < 28; ++y) {
                    for (size_t x = 0; x < 28; ++x) {
                    unsigned int srcX = x * capturedImg.getSize().x / 28;
                    unsigned int srcY = y * capturedImg.getSize().y / 28;

                    sf::Color color = capturedImg.getPixel(srcX, srcY);
                    double grayValue = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
                    
                    grayValue = (grayValue > 128 ? 1.0 : 0.0);
                    imageMatrix(y * 28 + x, 0) = grayValue;
                    }
                }
                imageMatrix(784, 0) = 1; 

                numberText.setString(std::to_string(lg.get_prediction(imageMatrix)));       
            }
        }

        window.draw(brushField);
        window.draw(clearButton);
        window.draw(clearButtonText);
        
        window.draw(numberField);
        window.draw(numberText);
        window.display();
    }
}
