#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_POINTS 100
#define EPOCHS 1000
#define LEARNING_RATE 0.01
#define C_PARAM 9.0

typedef struct {
    double x[2];
    int label; // +1 or -1
} Point;

int main() {
    srand(time(NULL));
    Point data[NUM_POINTS];

    // 1. Generate Synthetic Data separated by the line: x1 = 2*x0 + 1
    for (int i = 0; i < NUM_POINTS; i++) {
        data[i].x[0] = ((double)rand() / RAND_MAX) * 10 - 5; // Range [-5, 5]
        data[i].x[1] = ((double)rand() / RAND_MAX) * 10 - 5; // Range [-5, 5]
        
        double line_val = 2 * data[i].x[0] + 1;
        
        double noise = (((double)rand() / RAND_MAX) * 2.0) - 1.0; 
        
        if (data[i].x[1] > line_val + noise) {
            data[i].label = 1;
        } else {
            data[i].label = -1;
        }
    }

    // 2. Initialize Weights and Bias
    double w[2] = {0.0, 0.0};
    double b = 0.0;

    // 3. Train Soft-Margin SVM using Gradient Descent
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < NUM_POINTS; i++) {
            // Calculate functional margin: y * (w \cdot x + b)
            double decision = data[i].label * (w[0] * data[i].x[0] + w[1] * data[i].x[1] + b);

            // Condition for Hinge Loss gradient
            if (decision < 1.0) {
                // Misclassified or within the margin -> update weights and bias
                w[0] = w[0] - LEARNING_RATE * (w[0] / C_PARAM - data[i].label * data[i].x[0]);
                w[1] = w[1] - LEARNING_RATE * (w[1] / C_PARAM - data[i].label * data[i].x[1]);
                b = b + LEARNING_RATE * data[i].label * C_PARAM;
            } else {
                // Correctly classified outside the margin -> apply regularizer only
                w[0] = w[0] - LEARNING_RATE * (w[0] / C_PARAM);
                w[1] = w[1] - LEARNING_RATE * (w[1] / C_PARAM);
            }
        }
    }

    system("cls");
    printf("--- Training Complete ---\n");
    printf("Learned Hyperplane Weights: w0 = %.4f, w1 = %.4f\n", w[0], w[1]);
    printf("Learned Bias: b = %.4f\n", b);
    printf("Estimated Decision Line: x1 = %.2fx + %.2f\n", -w[0]/w[1], -b/w[1]);
    printf("Original Target Line:  x1 = 2.00x + 1.00\n");

    return 0;
}