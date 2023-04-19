#-------------------------------------------------------------------------
# AUTHOR: Mason Nash
# FILENAME: bagging_random_forest.py
# SPECIFICATION: optimize performance of a random_forest model
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
y_training = []
X_test = []
y_test = []
classVotes = [] #this array will be used to count the votes of each classifier

# code to read a contact_lens csv into a 2d list
def readCSVToList(dsIN):
    dbOut = []
    with open(dsIN, 'r') as csvfile:
       reader = csv.reader(csvfile)
       for i, row in enumerate(reader):
           if i > 0:  # skipping the header
               #debug
               dbOut.append(row)
    
    #print (dbOut)
    return dbOut


#reading the training data from a csv file and populate dbTraining
#--> add your Python code here
dbTraining = readCSVToList('optdigits.tra')

#reading the test data from a csv file and populate dbTest
#--> add your Python code here
dbTest = readCSVToList('optdigits.tes')

# split dbTest into feature list and class list 
for i, testSample in enumerate(dbTest):
   X_test.append(testSample[0 : len(testSample) - 1])
   y_test.append(testSample[len(testSample) - 1])


#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here
# initialize classVotes
classVotes = [ [0]*10 for i in y_test]


print("Started my base and ensemble classifier ...")

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

  bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

  #populate the values of X_training and y_training by using the bootstrapSample
  #--> add your Python code here
  for sample in bootstrapSample :
     X_training.append(sample[0:len(sample)-1])
     y_training.append(sample[len(sample)-1])
   
  #fitting the decision tree to the data
  clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
  clf = clf.fit(X_training, y_training)

   
  correctPredictions = 0
  totalPredictions = 0

  for (testSample, trueClass, voteRow) in zip(X_test, y_test, classVotes):

      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      #--> add your Python code here
      prediction = clf.predict([testSample])[0]
      voteRow[int(prediction)] += 1

      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
         if prediction == trueClass : # compare prediciton to last element of corresponding row of dbtest
               correctPredictions += 1

  if k == 0: #for only the first base classifier, print its accuracy here  #--> add your Python code here
      accuracy = correctPredictions/len(y_test)
      print("Finished my base classifier (fast but relatively low accuracy) ...")
      print("My base classifier accuracy: " + str(accuracy))
      print("")

#now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
#--> add your Python code here
correctPredicitons = 0
for trueClass, voteRow in zip(y_test, classVotes) :
   # find ensemble prediction
   highestTally = 0
   prediction = -1
   for i, tally in enumerate(voteRow):
      if tally > highestTally :
         highestTally = tally
         prediction = i
   
   # check if prediciton is right
   if prediction == int(trueClass) :
       correctPredictions += 1
# calculate accuracy of ensemble classifier
print(correctPredicitons)
print(len(y_test))
accuracy = correctPredictions/len(y_test)
        

#printing the ensemble accuracy here
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(accuracy))
print("")

print("Started Random Forest algorithm ...")

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

#Fit Random Forest to the training data
clf.fit(X_training,y_training)

#make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
#--> add your Python code here

classes_predicted_rf = clf.predict(X_test)

#compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
#--> add your Python code here
correctPredictions = 0
for samplePrediction, trueClass in zip(classes_predicted_rf, y_test) :
   if samplePrediction == trueClass :
      correctPredictions += 1

#calculate accuracy for rf
accuracy = correctPredictions / len(y_test)

#printing Random Forest accuracy here
print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
