#  Outline

Goal: To determine a methodlogy for composing a Neural Network for a specific task. From Linear Regression, there is a model composition algorithm called "Stepwise Regression". This process tries all possible combinations of variables to test and come up with the smallest model that results in the largest success. 


## Stepwise Regression
The basis for the algorithm is to start with single variables, test each one by itself in order to determine which of the variables are significant in fitting the data; whether or not the variable is useful in making the predictions. Based on this output, we can then build up on top of this model and slowly converge upon the optimal variable combination, and stopping when we go below our given threshold or "heuristic" for adding terms or combination of terms to the model. 

For example :
    
    y = ax + z
    y = bx + z
    y = cx + z
Lets say that we see that b is the most signinificant value for predictions, or if you had to choose a single variable to fit the data, b would be the best. Then we can continue testing each other permutation with the variables, stopping when we are no longer actively improving the model. For example: 

    a,b
    a,c
    a,a^2
    a,b^2
    a,c^2
    a,b,c
    a,b,c,a^2
    a,b,c,b^2
    a,b,c,c^2
    a,b,c,ab
    a,b,c,ac
    a,b,c,bc
    
So on and so forth, stopping when the new added variable combination no longer meaningfully adds to the model. This algorithm was the basis for the testing process of the NN training. The concept transfers over incredibly well to the composition of the the layers of the NN. 

## Types of Layers 
For this level of image classification, we chose to limit ourselves to Dense, Convolutional, Pooling, and Flattening. 

By slowly building and testing the combinations of layers, we can compose the most optimal model in order to reduce unecessary computations. As opposed to smaller projects, to run 5 individual training sessions of all 10,220 images takes roughly 8 hours after being accelerated by a GPU through CudaNN. So removing uneccessary hidden layers is very desriable.

## Development Process

Spinal Analysis

Cats and Dogs

Preprocessing Data

Dogs and Dogs

Gpu Acceleration 

Much better Dogs and Dogs

Future: 

Live Image recognition

VGG16 implementation 

UI

Spinal Analysis For 699 Next semester

## Tools 
1. Python 3 
2. Tensorflow-gpu
3. Keras
4. Kaggle
5. CudaNN
6. Anaconda
7. Python within Python