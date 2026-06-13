#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

/*  This program implements a simple model to predict used car prices based on the purchase price, age (years), 
   and mileage. The model uses three weights (w0, w1, w2) in an exponential model to approximate the relationship between 
   these input features and the estimated car price. The application generates synthetic data points using a true underlying 
   model with known weights and then trains the model using gradient descent to minimize the prediction error. During training, 
   the model adjusts its weights iteratively based on the difference between the predicted price and the generated estimated price. 
   After training, the model can predict the price of a car for given input values, providing an estimated resale value. 
   The algorithm demonstrates supervised learning with a nonlinear model and weight optimization through iterative updates.
*/

class CarPriceModel{
    public: 
        long double w0, w1, w2;
        
        CarPriceModel(long double w0 = 0.5, long double w1 = 0.5, long double w2 = 0.5)
        : w0(w0), w1(w1), w2(w2){}
        
        long double predictScaled(double price, double yearOld, double mileage){
            double price_scaled = price / 300000.0;
            double yearOld_scaled = yearOld / 30.0;
            double mileage_scaled = mileage / 300000.0;

            return w0 * price_scaled * std::exp(w1 * (-yearOld_scaled)) * std::exp(w2 * (-mileage_scaled));
        }
        
        long double predict(double price, double yearOld, double mileage){
            return predictScaled(price, yearOld, mileage) * 300000.0;
        }
};

class DataPoint{
    public: 
        double boughtPrice, yearOld, milage, estimatedPrice;
        DataPoint(double boughtPrice = 0, double yearOld = 0, double milage = 0, double estimatedPrice = 0)
        : boughtPrice(boughtPrice), yearOld(yearOld), milage(milage), estimatedPrice(estimatedPrice){}
};

class ModelTrainer{
    private: 
        CarPriceModel& carPriceModel;
        std::vector<DataPoint> dataPoints;
        double alpha = 0.05; 
        double meanAbsoluteError = 0;

        long double getRandomDouble(long double min, long double max){
            return min + (max - min) * (rand() / (double)RAND_MAX);
        }

        void createDataPoints(){
            long double w0_true = 1.0;
            long double w1_true = 3.2;
            long double w2_true = 0.08;

            for(int i = 0; i < 2000; i++){
                long double boughtPrice = getRandomDouble(10000, 300000);
                long double yearOld = getRandomDouble(0, 30);
                long double milage = getRandomDouble(0, 300000);

                long double estimatedPrice = w0_true * (boughtPrice / 300000.0) * std::exp(w1_true * (-yearOld / 30.0)) * std::exp(w2_true * (-milage / 300000.0));

                dataPoints.emplace_back(boughtPrice, yearOld, milage, estimatedPrice);
            }
        }

        void train(int epoch){
            for (int round = 0; round < epoch; round++){
                for (auto & dp : dataPoints){
                    double price_scaled = dp.boughtPrice / 300000.0;
                    double year_scaled = dp.yearOld / 30.0;
                    double mileage_scaled = dp.milage / 300000.0;

                    double predictedPriceScaled = carPriceModel.predictScaled(dp.boughtPrice, dp.yearOld, dp.milage);
                    double error = predictedPriceScaled - dp.estimatedPrice;

                    carPriceModel.w0 -= alpha * error * (price_scaled * std::exp(-carPriceModel.w1 * year_scaled) * std::exp(-carPriceModel.w2 * mileage_scaled));
                    carPriceModel.w1 -= alpha * error * (-year_scaled * predictedPriceScaled);
                    carPriceModel.w2 -= alpha * error * (-mileage_scaled * predictedPriceScaled);
                }
            }
        }

    public:
        ModelTrainer(CarPriceModel& carPriceModel) : carPriceModel(carPriceModel){}
        
        void evaluate(){
            double totalError = 0;
            for (auto& dp : dataPoints){
                totalError += std::abs((dp.estimatedPrice * 300000.0) - carPriceModel.predict(dp.boughtPrice, dp.yearOld, dp.milage));
            }
            meanAbsoluteError = totalError / dataPoints.size();
        }

        void run(){
            createDataPoints();
            train(100);
            evaluate();

            std::cout << "The equation from which the synthetic data was created is:\n";
            std::cout << "w0_true * boughtPrice/300000 * exp(w1_true * (-yearOld/30)) * exp(w2_true * (-mileage/300000))\n";
            std::cout << "The weights of the synthetic data are:\n";
            std::cout << "w0 = 1, w1 = 3.2, w2 = 0.08\n\n";

            std::cout << "The model's weights after training are: W0 = " << carPriceModel.w0 << ", W1 = " << carPriceModel.w1 << ", W2 = " << carPriceModel.w2 << "\n";
            std::cout << "The Mean Absolute Error is: $" << meanAbsoluteError << "\n";
        }
};

int main(){
    system("cls");
    std::srand(std::time(nullptr));
    
    CarPriceModel carPriceModel;
    ModelTrainer mt(carPriceModel);
    mt.run();

    return 0;
}