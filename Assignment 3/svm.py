#-------------------------------------------------------------------------
# AUTHOR: Mason nash
# FILENAME: svm.py
# SPECIFICATION: finds the optimal hyperparameters of an SVM model to identify handwritten digits.
# FOR: CS 4210- Assignment #3
# TIME SPENT: time on just the code: 50 mins (not counting the lengthy break i took to watch School of Rock.)
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

# remember best performance
maxAccuracy = 0
# iterate c
for cInst in c :
    
    # iterate degree
    for dInst in degree :
        # iterate kernel
        for kInst in kernel :
            #  iterate decision_function_shape
            
            for shapeInst in decision_function_shape :
                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                #--> add your Python code here

                classifier = svm.SVC(C=cInst, degree=dInst, kernel=kInst, decision_function_shape =shapeInst)

                #Fit SVM to the training data
                #--> add your Python code here
                classifier.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                #--> add your Python code here

                countCorrect = 0
                for (x_testSample, y_testSample) in zip(X_test, y_test):

                    if (classifier.predict([x_testSample]) == y_testSample) :
                        countCorrect += 1
                    
                # calculate accuracy
                accuracy = (countCorrect / y_test.size)



                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                
                if accuracy > maxAccuracy :
                    maxAccuracy = accuracy
                    print (f'Highest SVM accuracy so far: {maxAccuracy}, Parameters: C={cInst}, degree={dInst}, kernel={kInst}, decision_function_shape ={shapeInst}')




