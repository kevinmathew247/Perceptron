import numpy as np
import pandas as pd
import warnings

L2Regularised = False
maximumIterations = 20
#loading the dataset
train_data = pd.read_csv("train.data", header=None)
test_data = pd.read_csv("test.data", header=None)
warnings.filterwarnings("ignore")

"""
myPerceptron(X_train, y_train, maxIterations) - trains the perceptron model with the training data and 
target label as the input along with the maximum number of iterations it has to be run
"""
def myPerceptronTrain(X_train, y_train, maxIterations):
    feature_length = X_train.shape[1]
    
    # set initial bias as 0
    b = 0
    # set initial weights as zero for all inputs
    w = np.zeros(feature_length)
    for iter in range(maxIterations):
        # zip function is used to create tuples of objects X_train and y_train
        for x, y in zip(X_train, y_train):
            #finding activation score
            a = np.dot(x, w) + b
            predicted_y = activation_func(a)
            for weight in range(len(w)):
                if y*a <= 0:
                    # move on to the update step if y*a <= 0
                    b = b+y
                    w[weight] += y * x[weight]
    return w, b

def activation_func(x):
        return np.where(x >= 0, 1, 0)    


"""
myPerceptronTest(testFeatures, w, b) - this function is to test the perceptron model with 
the test data using the weights and bias obtained while training the model
"""
def myPerceptronTest(testFeatures, w, b):
    activationScore = np.zeros(len(testFeatures))
    for i in range(len(testFeatures)):
        # finding activation score
        activationScore[i] = np.dot(testFeatures[i], w) + b    
    return np.sign(activationScore)


"""
findAccuracy(trueLabels, predLabels) - this function is to find the accuracy between the 
true/actual labels and the predicted labels
"""
def findAccuracy(trueLabels, predLabels):
    y_true = trueLabels
    y_pred = predLabels
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
    return accuracy


"""
l2Regularisation(xTrain, yTrain, maxIterations) - this function is to find the accuracy of the model
after adding an L2 regularisation term. The regularisation coefficients were provided.
"""
def l2Regularisation(xTrain, yTrain, maxIterations):
    featureValue = xTrain.shape[1]
    w = np.zeros(featureValue)
    regBias = 0
    # initialising the regularisation coefficients provided
    regCoefficients = [0.01, 0.1, 1.0, 10.0, 100.0]
    for lamda in regCoefficients:
        for iter in range(maxIterations):
            for x, y in zip(xTrain, yTrain):
                a = np.dot(x, w) + regBias
                for obj in range(len(w)):
                    if y*a <= 0:
                        # if y*a <= 0, update the weights and the bias
                        w[obj] = (((1-2*lamda)) * w[obj]) + (x[obj] * y)
                        regBias = regBias+y
    return w, regBias


"""
oneVsOneClassification(c1Label, c2Label) - this function is to find the accuracies obtained using
one vs one binary classification between any two classes
"""
def oneVsOneClassification(c1Label, c2Label):
    trainDataSet = train_data[(train_data[4] == c1Label) | (train_data[4] == c2Label)]
    testDataSet = test_data[(test_data[4] == c1Label) | (test_data[4] == c2Label)]

    trainData = trainDataSet.iloc[:, :-1].values
    trainLabels = trainDataSet.iloc[:, -1].values

    testData = testDataSet.iloc[:, :-1].values
    testLabels = testDataSet.iloc[:, -1].values

    trainLabels[trainLabels == c1Label] = 1
    trainLabels[trainLabels == c2Label] = -1
    testLabels[testLabels == c1Label] = 1
    testLabels[testLabels == c2Label] = -1

    w, b = myPerceptronTrain(trainData, trainLabels, 20)
    predTestValues = myPerceptronTest(testData, w, b)
    predTrainValues = myPerceptronTest(trainData, w, b)
    testAccuracy = findAccuracy(testLabels, predTestValues)
    trainAccuracy = findAccuracy(trainLabels, predTrainValues)

    print("\nTesting accuracy for {} vs {} = {}".format(c1Label,c2Label,testAccuracy))
    print("Training accuracy for {} vs {} = {}".format(c1Label,c2Label,trainAccuracy))


"""
oneVsRestClassification(cLabel) - this function is to find the accuracies obtained using
one vs rest binary classification between any one class and the rest of the classes.
This function also outputs the accuracies for all questions.
"""
def oneVsRestClassification(cLabel):
    #slicing the datasets to obtain required values
    yTrain = train_data.iloc[:, -1].values
    xTrain = train_data.iloc[:, :-1].values
    yTest = test_data.iloc[:, -1].values
    xTest = test_data.iloc[:, :-1].values

    #Assigning 1 to the input class label and -1 to all the other class labels
    #in both training and test datasets
    yTrain[yTrain == cLabel] = 1
    yTrain[yTrain != 1] = -1
    yTest[yTest == cLabel] = 1
    yTest[yTest != 1] = -1

    if not L2Regularised:
        w, b = myPerceptronTrain(xTrain, yTrain, maximumIterations)
        predictedTestLabels = myPerceptronTest(xTest, w, b)
        predictedTrainLabels = myPerceptronTest(xTrain, w, b)
        testAccuracy = findAccuracy(yTest, predictedTestLabels)
        trainAccuracy = findAccuracy(yTrain, predictedTrainLabels)
        print("\nTesting accuracy for {} vs rest = {}".format(cLabel, testAccuracy))
        print("Training accuracy for {} vs rest = {}".format(cLabel, trainAccuracy))
    else:
        w, b = l2Regularisation(xTrain, yTrain, maximumIterations)
        predictedTestLabels = myPerceptronTest(xTest, w, b)
        predictedTrainLabels = myPerceptronTest(xTrain, w, b)
        testAccuracy = findAccuracy(yTest, predictedTestLabels)
        trainAccuracy = findAccuracy(yTrain, predictedTrainLabels)
        print("\nTesting accuracy for {} vs rest = {}".format(cLabel, testAccuracy))
        print("Training accuracy for {} vs rest = {}".format(cLabel, trainAccuracy))
        
#print the outputs of one vs one classifications
print("One vs One Classification")
oneTwoClass = oneVsOneClassification('class-1', 'class-2')
twoThreeClass = oneVsOneClassification('class-2', 'class-3')
oneThreeClass = oneVsOneClassification('class-1', 'class-3')

#print the outputs of one vs rest classifications
print("\n\n\nOne vs Rest Classification")
oneRestClass = oneVsRestClassification('class-1')
twoRestClass = oneVsRestClassification('class-2')
threeRestClass = oneVsRestClassification('class-3')

#print the outputs of one vs rest classifications after adding L2 regularisation term
print("\n\n\nOne vs Rest Classification after adding L2 regularisation term")
#setting the value of L2Regularised as true before passing it to the function
L2Regularised = True
oneRestL2 = oneVsRestClassification('class-1')
twoRestL2 = oneVsRestClassification('class-2')
threeRestL2 = oneVsRestClassification('class-3')