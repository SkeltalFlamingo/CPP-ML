#-------------------------------------------------------------------------
# AUTHOR: Mason Nash
# FILENAME: decision_tree_2.py
# SPECIFICATION: makes decision trees from several csv files, then compares their performance
# FOR: CS 4210- Assignment #2
# TIME SPENT: this file took 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv


# code to transform db features and classes into numbers
def transformDB(db):
    Classes = []
    Features = []
    
    # transform features
    for i, dbRow in enumerate(db):

        Features.append([0, 0, 0, 0])
  
        # age parsing
        col = 0
        if db[i][col] == "Young":
            Features[i][col] = 1
        elif db[i][col] == "Prepresbyopic":
            Features[i][col] = 2
        elif db[i][col] == "Presbyopic":
            Features[i][col] = 3

        # Spectacle Parsing
        col = 1
        if db[i][col] == "Myope":
            Features[i][col] = 1
        elif db[i][col] == "Hypermetrope":
            Features[i][col] = 2

        # Astigmatism parsing
        col = 2
        if db[i][col] == "No":
            Features[i][col] = 1
        elif db[i][col] == "Yes":
            Features[i][col] = 2

        # Tear Production parsing
        col = 3
        if db[i][col] == "Reduced":
            Features[i][col] = 1
        elif db[i][col] == "Normal":
            Features[i][col] = 2
        
    # transform classes
    for row in db:
        if row[4] == "Yes":
            Classes.append(1)
        elif row[4] == "No":
            Classes.append(2)
        else:
            Classes.append(0)
    
    return Features, Classes

# code to read a contact_lens csv into a 2d list
def readCSVtoList(dsIN):
    dbOut = []
    with open(dsIN, 'r') as csvfile:
       reader = csv.reader(csvfile)
       for i, row in enumerate(reader):
           if i > 0:  # skipping the header
               #debug
               dbOut.append(row)
    
    #print (dbOut)
    return dbOut


############# PROGRAM START ###################################################################################################################

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']


for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    dbTraining = readCSVtoList(ds)
    
    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =

    ########### SEE THE NEXT PART FOR Y. I COMBINED THESE FEATURES INTO ONE FUNCTION.
    
    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    
    X, Y = transformDB(dbTraining)
    
    
    
    #loop your training and test tasks 10 times here
    accuracySum = float(0)
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = readCSVtoList("contact_lens_test.csv")
        #print(dbTest)

        
        #transform the features of the test instances to numbers following the same strategy done during training,
        #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        #--> add your Python code here
            
        XTest, YTest = transformDB(dbTest)
        #for testSample in XTest:
            #print(testSample)
        
        # try predicting each test sample
        correctPredictions = 0
        for (testSample, trueClass) in zip(XTest, YTest):
            class_predicted = clf.predict([testSample])[0]
        
            # accuracy = correct / total
            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if class_predicted == trueClass :
                correctPredictions += 1
                
        accuracy = correctPredictions/len(YTest)
        #print(f"        accuracy on itt {i} is {accuracy}")
        
        accuracySum += accuracy

    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    totalAccuracy = accuracySum/10

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"final accuracy when training on {ds}: {totalAccuracy}")



