# my-ai-learning-journey-from-scratch
A self-taught deep dive into the foundations of AI. Documenting my journey of building algorithms from scratch to understand the math and mechanics behind how AI actually learns.
Most of my projects will be in C++

Some simple projets:
1. Linear Decision Boundary
    - **Project Goal:** The algorithm learns whether the sum of two numbers is greater than or less than 100 using supervised learning.
    - **Classification Rules:** 
        - 1 -> (x1 + x2 > 100) 
        - 0 -> (x1 + x2 < 100)
    - **Implementation Details:** It implements a single-layer Perceptron from scratch to find the optimal weights and bias for a linear decision boundary. The model is trained on a synthetic dataset of 10,000 generated coordinates $(x_1, x_2)$, where each point is labeled based on whether its sum exceeds a threshold of 100 ($x_1 + x_2 > 100$). By utilizing a basic threshold activation function and adjusting the weights ($w_1, w_2, w_3$) incrementally through supervised learning steps governed by a learning rate ($\alpha = 0.0005$), the algorithm iteratively minimizes classification errors over 40 epochs. Ultimately, the code successfully learns to approximate the mathematical boundary dividing the two data classes.
2. sdf