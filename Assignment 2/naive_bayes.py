#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv


def transformDB(db):
    Classes = []
    Features = []

    # transform features
    for i, dbRow in enumerate(db):

        Features.append([0, 0, 0, 0])

        # Outlook parsing
        col = 1     # start at one to eliminate Day
        if db[i][col] == "Sunny":
            Features[i][col -1] = 1
        elif db[i][col -1] == "Overcast":
            Features[i][col -1] = 2
        elif db[i][col] == "Rain":
            Features[i][col - 1] = 3

        # Temperature Parsing
        col = 2
        if db[i][col] == "Mild":
            Features[i][col - 1] = 1
        elif db[i][col] == "Cool":
            Features[i][col - 1] = 2
        elif db[i][col] == "Hot":
            Features[i][col - 1] = 3


        # Humidity parsing
        col = 3
        if db[i][col] == "High":
            Features[i][col - 1] = 1
        elif db[i][col] == "Normal":
            Features[i][col - 1] = 2

        # Wind parsing
        col = 4
        if db[i][col] == "Weak":
            Features[i][col - 1] = 1
        elif db[i][col] == "Strong":
            Features[i][col - 1] = 2

    # transform classes
    for row in db:
        if row[5] == "Yes":
            Classes.append(1)
        elif row[5] == "No":
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
               #print(row)
               dbOut.append(row)

    return dbOut

############   PROGRAM START    ###########################

#reading the training data in a csv file
#--> add your Python code here

trainingDB = readCSVtoList("weather_training.csv")

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here


X, Y = transformDB(trainingDB)


#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

##### SEE LINE 95 FOR INITIALIZING Y #############

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)
#reading the test data in a csv file
#--> add your Python code here

testDB = readCSVtoList("weather_test.csv")
XTest, YTest = transformDB(testDB)

#printing the header os the solution
#--> add your Python code here
print("Day    Outlook    Temperature   Humidity   Wind    PlayTennis   Confidence ")

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for sample, rawSample in zip(XTest, testDB) :
    probability = clf.predict_proba([sample])[0]
    
    #find higher likelyhood class
    mostLikelyClass = ""
    probIndex = -1
    if probability[0] > probability[1] :
        mostLikelyClass = "Yes"
        probIndex = 0
    else :
        mostLikelyClass = "No"
        probIndex = 1
    
    if probability[probIndex] >= 0.75 :
        
        print("{:6s} {:10s} {:13s} {:10s} {:7s} {:9s} {:7.2f}".format(
            rawSample[0], rawSample[1], rawSample[2], rawSample[3], rawSample[4], mostLikelyClass, probability[probIndex]
            ))

