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
        std::vector<double> input{2};
        std::vector<double> hidden{2};  
        std::vector<double> output{1};  

        std::vector<std::vector<double>> WIH{ 2, std::vector<double>(2)};
        std::vector<double>WHO{2};

        std::vector<double> biasHidden{2};
        std::vector<double> biasOutput{1};

        double learningRateAlpha = 0.5;

        // XOR Training dataset
        std::vector<std::vector<double>> X = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        std::vector<double> Y = {0, 1, 1, 0};

        NeuralNetwork() {
            getSomeRandomWeights();
            train(10000);
            evaluate();

        }

        void getSomeRandomWeights(){
            srand(time(0));
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++)
                    WIH[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
            for (int i = 0; i < 2; i++)
                WHO[i] = ((double)rand() / RAND_MAX) * 2 - 1;

            for (int i = 0; i < 2; i++)
                biasHidden[i] = ((double)rand() / RAND_MAX) * 2 - 1;

            biasOutput[0] = ((double)rand() / RAND_MAX) * 2 - 1;
        }

        void forward() {
            for (int i = 0; i < 2; i++) {
                hidden[i] = sigmoid(input[0] * WIH[0][i] +
                                    input[1] * WIH[1][i] +
                                    biasHidden[i]);
            }

            output[0] = sigmoid(hidden[0] * WHO[0] +
                                hidden[1] * WHO[1] +
                                biasOutput[0]);
        }

        void train( int epochs) {
            for (int e = 0; e < epochs; e++) {
                double total_error = 0.0;

                for (size_t s = 0; s < X.size(); s++) {

                    input = X[s];
                    forward();

                    double error = Y[s] - output[0];
                    total_error += error * error;

                    double output_gradient = error * sigmoid_derivative(output[0]);

                    double hidden_gradient[2];
                    for (int i = 0; i < 2; i++) {
                        hidden_gradient[i] = output_gradient * WHO[i] *
                                    sigmoid_derivative(hidden[i]);
                    }

                    // Update weights hidden → output
                    for (int i = 0; i < 2; i++) {
                        WHO[i] += learningRateAlpha * output_gradient * hidden[i];
                    }
                    biasOutput[0] += learningRateAlpha * output_gradient;

                    // Update weights input → hidden 
                    for (int i = 0; i < 2; i++) {
                        for (int j = 0; j < 2; j++) {
                            WIH[i][j] += learningRateAlpha * hidden_gradient[j] * input[i];
                        }
                    }
                    for (int i = 0; i < 2; i++) {
                        biasHidden[i] += learningRateAlpha * hidden_gradient[i];
                    }
                }

                if (e % 1000 == 0)
                    std::cout << "Epoch " << e << " Error: " << total_error << "\n";
            }
        }
        void evaluate(){
            for (size_t i = 0; i < X.size(); i++) {
                input= X[i];
                forward();
                std::cout << X[i][0] << " XOR " << X[i][1]
                        << " = " << output[0] << "\n";
            }
        }
};

int main() {
    NeuralNetwork nn;

}
