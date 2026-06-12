    #include <iostream>
    #include <vector>
    #include <cstdlib>
    #include <ctime>
    /* This algorithem is learning if the sum of two numbers is greater or lower then 100
        we generatet 10000 data points that helps the algorithem create the boundury ( the line ) between the 
        result  1 ->( x1+x2 >100) and  0 ->( x1+x2<100)
    */
    class  Ai{
        public:
            double w1, w2 ,w3 ;
            double alpha=0.0005;
            Ai(double w1=0,double w2=0,double w3=0):w1(w1),w2(w2),w3(w3){}
            void printWeights() {
                std::cout << "\nWeights:\n";
                std::cout << "w1 = " << w1 << ", w2 = " << w2 << ", w3 = " << w3 << "\n";
            }
    };
    class DataPoint{
        public:
            int x1, x2;
            int y;
            DataPoint(int x1 ,int x2, int y): x1(x1) , x2(x2) , y(y){}
    };
    class Application{
        private:
            Ai& ai;
            std::vector<DataPoint> datapoints ;
            void createDataPoints(){
                for ( int i =0; i < 10000 ; i++){
                    int rand1 = std::rand() % 100 + 1; 
                    int rand2 = std::rand() % 100 + 1; 
                    int y=((rand1 +rand2)>100) ? 1 : 0 ; 
                    datapoints.push_back(DataPoint(rand1, rand2,y));
                }
            }
            void train(int epochs){
                createDataPoints();
                for (int e = 0; e < epochs; ++e) {
                    for(const DataPoint& a : datapoints){
                        int y =predict(a.x1,a.x2);
                        
                            ai.w1+=ai.alpha*(a.y-y)*1;
                            ai.w2+=ai.alpha*(a.y-y)*a.x1;
                            ai.w3+=ai.alpha*(a.y-y)*a.x2;
                        
                    }
                }
            }
        public : 
            int predict(int x1 ,int x2){
                double y= ai.w1 + ai.w2*x1 + ai.w3*x2;
                int result=(y >= 0) ? 1 : 0;
                return result;
            }
            void test(int x1, int x2){
                double y= ai.w1 + ai.w2*x1 + ai.w3*x2;  
                std::cout<<"For x1= "<<x1<<", x2="<<x2<< " , y is  :"<< y;
                int result=predict(x1, x2);
                std::cout<<"\nResult is :"<<result;
            }
            void evaluate(int numSamples = 1000) {
                int correct = 0;

                for (int i = 0; i < numSamples; ++i) {
                    int x1 = std::rand() % 100 + 1;
                    int x2 = std::rand() % 100 + 1;
                    int actual = (x1 + x2 > 100) ? 1 : 0;
                    int predicted = predict(x1, x2);

                    if (predicted == actual) {
                        correct++;
                    }
                }

                double accuracy = (double)correct / numSamples * 100;
                std::cout << "Accuracy on " << numSamples << " test samples: " << accuracy << "%\n";
            }
            Application(Ai &ai): ai(ai){}

            void run(){
                train(40);  // for each datapoint it trains it 40 times
                test(94,7);
            }
    };
    int main(){
        std::srand(std::time(nullptr));  
        Ai ai(0,0,0);
        Application app(ai);
        app.run();
        ai.printWeights();
        app.evaluate();
        return 0 ;
    }