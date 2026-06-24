#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>

struct IrisSample{
    std::vector<double> features;
    std::vector<double> target;
};

double sigmoid_function(double z){
    return 1.0 / ( 1.0 + exp(-z) );
}

double sigmoid_derivative(double x ){
    return x * ( 1 - x );
}

class NeuralNetwork{

    private:

        std::vector<double> inputNeuron{std::vector<double>(4)};   
        std::vector<double> hiddenNeuron{std::vector<double>(14)};  
        std::vector<double> outputNeuron{std::vector<double>(3)};

        std::vector<std::vector<double>> WIH{ 4 , std::vector<double>(14) };
        std::vector<std::vector<double>> WHO{ 14 , std::vector<double>(3) };

        std::vector<double> biasHidden {std::vector<double>(14)};
        std::vector<double> biasOutput {std::vector<double>(3)};

        double learningRateAlpha = 0.2;
        std::vector<IrisSample> dataset;

        void getSomeRandomWeights(){

            for ( int i = 0 ; i < 4 ; i ++ ){
                for ( int j = 0 ; j < 14 ; j++){
                    WIH[i][j]=((double)rand()/RAND_MAX) * 2 - 1 ;
                }
            }
            for ( int i = 0 ; i < 14 ; i ++ ){
                for ( int j = 0 ; j < 3 ; j++){
                    WHO[i][j]=((double)rand()/RAND_MAX) * 2 - 1 ;
                }
            }

            for ( int i =0 ; i < 14 ; i++ ){
                biasHidden[i]=((double)rand()/RAND_MAX) * 2 - 1 ;
            }
            for ( int i =0 ; i < 3 ; i++ ){
                biasOutput[i]=((double)rand()/RAND_MAX) * 2 - 1 ;
            }
        }

        void loadIrisDataset(const std::string& filename){

            std::ifstream file(filename);
            if(!file.is_open()){
                std::cerr<<"Error in opening file";
                return;
            }

            std::string line;
            while(std::getline(file , line)){

                if(line.empty()) continue;

                std::stringstream ss(line);
                std::string value;
                IrisSample sample;

                for (int i = 0; i < 4; ++i) {
                    if (std::getline(ss, value, ',')) {
                        sample.features.push_back(std::stod(value)); 
                    }
                }

                // Read the label
                if (std::getline(ss, value, ',')) {
                    if (value == "Iris-setosa") {
                        sample.target = {1.0, 0.0, 0.0};
                    } 
                    else if (value == "Iris-versicolor") {
                        sample.target = {0.0, 1.0, 0.0};
                    } 
                    else if (value == "Iris-virginica") {
                        sample.target = {0.0, 0.0, 1.0};
                    }
                }
                dataset.push_back(sample);

            }
            file.close();
        }

        void train( int epochs){
            for ( int rounds =0; rounds< epochs ; rounds++){
                for ( const auto& sample : dataset){

                    forward( sample.features[0] , sample.features[1] , sample.features[2] , sample.features[3] );
                    std::vector<double> outputError(3);
                    for( int i=0 ; i <3 ; i++){
                        outputError[i] = outputNeuron[i] - sample.target[i];
                    }
                    
                    std::vector<double> hiddenError(14,0.0);
                    for( int i =0 ; i < 14; i++){
                        double error_sum =0.0;
                        for( int j =0 ; j <3 ;j++){
                            error_sum += outputError[j]*WHO[i][j];
                        }
                        hiddenError[i] = error_sum * sigmoid_derivative(hiddenNeuron[i]);
                    }

                    // Update Weights and Biases between Hidden and Output Layer (WHO)
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 14; j++) {
                            WHO[j][i] -= learningRateAlpha * outputError[i] * hiddenNeuron[j];
                        }
                        biasOutput[i] -= learningRateAlpha * outputError[i];
                    }

                    // Update Weights and Biases between Input and Hidden Layer (WIH)
                    for (int i = 0; i < 14; i++) {
                        for (int j = 0; j < 4; j++) {
                            WIH[j][i] -= learningRateAlpha * hiddenError[i] * inputNeuron[j];
                        }
                        biasHidden[i] -= learningRateAlpha * hiddenError[i];
                    }
                }
            }
        }
    public:

        NeuralNetwork(){
            getSomeRandomWeights();
            loadIrisDataset("iris.csv");
            std::default_random_engine engine(time(0));
            std::shuffle(dataset.begin(), dataset.end(), engine);
            train(500);
        }

        
        void forward(double sepalLength , double sepalWidth , double petalLength , double petalWidth){
            inputNeuron[0]=sepalLength;
            inputNeuron[1]=sepalWidth;
            inputNeuron[2]=petalLength;
            inputNeuron[3]=petalWidth;

            for (int i =0 ; i < 14 ; i++){
                hiddenNeuron[i]=0;
                for ( int j =0 ; j < 4 ; j++){
                    hiddenNeuron[i] += WIH[j][i]*inputNeuron[j];
                }
                hiddenNeuron[i]+=biasHidden[i];
                hiddenNeuron[i] = sigmoid_function(hiddenNeuron[i]);
            }

            double sum_exp=0;
            for ( int i =0; i< 3 ; i++){
                outputNeuron[i]=0;
                for ( int j = 0 ; j< 14 ;j++){
                    outputNeuron[i] += WHO[j][i]*hiddenNeuron[j];
                }
                outputNeuron[i]+=biasOutput[i];
                outputNeuron[i] = std::exp(outputNeuron[i]);
                sum_exp += outputNeuron[i];
            }
            // using softmax
            for (int i = 0; i < 3; i++){
                outputNeuron[i] = outputNeuron[i] / sum_exp;
            }
        }

        void printOutput(){
            std::cout << "\n--- Network Predictions ---" << std::endl;
            std::cout << "The probability for Iris-Setosa:     " << (outputNeuron[0] * 100.0) << "%" << std::endl;
            std::cout << "The probability for Iris-Versicolor: " << (outputNeuron[1] * 100.0) << "%" << std::endl;
            std::cout << "The probability for Iris-Virginica:  " << (outputNeuron[2] * 100.0) << "%" << std::endl;        
        }

        void evaluate(){
            int  CountRightAnswers=0;

            for( const auto& sample : dataset){
                forward(sample.features[0],sample.features[1],sample.features[2],sample.features[3]);
                auto maxValueOfOutput= std::max_element(outputNeuron.begin(), outputNeuron.end());
                int answerOfOutput = std::distance(outputNeuron.begin() ,  maxValueOfOutput);

                auto maxValueOfSample= std::max_element(sample.target.begin(), sample.target.end());
                int answerOfSample = std::distance(sample.target.begin() , maxValueOfSample);
                if(answerOfOutput== answerOfSample){
                    CountRightAnswers++;
                }
            }
            double percentage = (static_cast<double>(CountRightAnswers) / dataset.size()) * 100.0;
            std::cout<<"\nResults:";
            std::cout << "\nNetwork Architecture: 4 Input Neurons | 14 Hidden Neurons | 3 Output Neurons";
            std::cout << "\nTrained on 150 shuffled samples over 500 epochs.\n";
            std::cout<<"Correct answers : "<< CountRightAnswers <<" out of "<< dataset.size() <<" | in Percantage : "<<percentage<<" %\n\n";
        }

};

int main(){
    srand(time(0));
    NeuralNetwork nn;
    system("cls");
    nn.evaluate();
    return 0;
}