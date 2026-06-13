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
2. Nonelinear Regression Carprice
    - **Project Goal:** The algorithm learns to predict the resale price of a used car based on three input features—purchase price, age, and mileage—using supervised non-linear regression.
    - **Mathematical Framework:** The model utilizes an exponential decay structure to mirror real-world automotive depreciation, ensuring that estimated asset valuations scale realistically and never drop below zero. It is expressed as:
      $$\Large predictedPrice = w_0 \cdot \left(\frac{price}{300000}\right) \cdot e^{-w_1 \cdot \left(\frac{year}{30}\right)} \cdot e^{-w_2 \cdot \left(\frac{mileage}{300000}\right)} \cdot 300000$$
    - **Implementation Details:** It implements a multi-variable non-linear regression model trained from scratch using gradient descent to minimize an L2 Squared Error loss function. The training pipeline automatically generates a synthetic dataset of 2,000 car records with known target parameters ($w_0 = 1.0, w_1 = 3.2, w_2 = 0.08$). To ensure mathematical stability and prevent gradient explosion, all features are normalized via feature scaling during execution. By calculating exact partial derivatives through the calculus Chain Rule and updating the weights ($w_0, w_1, w_2$) incrementally using a learning rate ($\alpha = 0.05$) over 100 epochs, the model successfully converges onto the true underlying parameters and minimizes Mean Absolute Error (MAE).