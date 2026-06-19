#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


double sigmoid_derivative(double x) {
    return x * (1 - x);  // here x is already sigmoid(x)
}

class NeuralNetwork {
public:
    std::vector<double> input;   // 2 inputs
    std::vector<double> hidden;  // 2 hidden
    std::vector<double> output;  // 1 output

    // Weights
    double w_input_hidden[2][2]; // 2x2 matrix
    double w_hidden_output[2];   // 2x1 matrix

    // Bias
    double b_hidden[2];
    double b_output;

    double learning_rate = 0.5;

    NeuralNetwork() {
        input.resize(2);
        hidden.resize(2);
        output.resize(1);

        // Random weights
        srand(time(0));
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++)
                w_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        for (int i = 0; i < 2; i++)
            w_hidden_output[i] = ((double)rand() / RAND_MAX) * 2 - 1;

        for (int i = 0; i < 2; i++)
            b_hidden[i] = ((double)rand() / RAND_MAX) * 2 - 1;

        b_output = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    // Forward pass
    void forward() {
        for (int i = 0; i < 2; i++) {
            hidden[i] = sigmoid(input[0] * w_input_hidden[0][i] +
                                input[1] * w_input_hidden[1][i] +
                                b_hidden[i]);
        }

        output[0] = sigmoid(hidden[0] * w_hidden_output[0] +
                            hidden[1] * w_hidden_output[1] +
                            b_output);
    }

    // Backpropagation
    void train(std::vector<std::vector<double>> X,
               std::vector<double> Y, int epochs) {
        for (int e = 0; e < epochs; e++) {
            double total_error = 0.0;

            for (size_t s = 0; s < X.size(); s++) {
                // Set inputs
                input = X[s];

                // Forward pass
                forward();

                // Error
                double error = Y[s] - output[0];
                total_error += error * error;

                // Backpropagation
                double d_output = error * sigmoid_derivative(output[0]);

                double d_hidden[2];
                for (int i = 0; i < 2; i++) {
                    d_hidden[i] = d_output * w_hidden_output[i] *
                                  sigmoid_derivative(hidden[i]);
                }

                // Update weights hidden → output
                for (int i = 0; i < 2; i++) {
                    w_hidden_output[i] += learning_rate * d_output * hidden[i];
                }
                b_output += learning_rate * d_output;

                // Update weights input → hidden 
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        w_input_hidden[i][j] += learning_rate * d_hidden[j] * input[i];
                    }
                }
                for (int i = 0; i < 2; i++) {
                    b_hidden[i] += learning_rate * d_hidden[i];
                }
            }

            if (e % 1000 == 0)
                std::cout << "Epoch " << e << " Error: " << total_error << "\n";
        }
    }
};

int main() {
    NeuralNetwork nn;

    // XOR dataset
    std::vector<std::vector<double>> X = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<double> Y = {0, 1, 1, 0};

    // Train
    nn.train(X, Y, 10000);

    // Test
    for (size_t i = 0; i < X.size(); i++) {
        nn.input = X[i];
        nn.forward();
        std::cout << X[i][0] << " XOR " << X[i][1]
                  << " = " << nn.output[0] << "\n";
    }
}
