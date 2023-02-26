#-------------------------------------------------------------------------
# AUTHOR: Mason Nash
# FILENAME: decision_tree.py
# SPECIFICATION: this prgram reads in data from a csv file, then creates and displays a dicision tree from it.
# FOR: CS 4210- Assignment #1
# TIME SPENT: (just the code) 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open("contact_lens.csv", 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
# X =pi
for i, dbRow in enumerate(db):
  
  X.append([0, 0, 0, 0])
  
  # age parsing
  col = 0
  if db[i][col] == "Young":
    X[i][col] = 1
  elif db[i][col] == "Prepresbyopic":
    X[i][col] = 2
  elif db[i][col] == "Presbyopic":
    X[i][col] = 3

  # Spectacle Parsing
  col = 1
  if db[i][col] == "Myope":
    X[i][col] = 1
  elif db[i][col] == "Hypermetrope":
    X[i][col] = 2

  # Astigmatism parsing
  col = 2
  if db[i][col] == "No":
    X[i][col] = 1
  elif db[i][col] == "Yes":
    X[i][col] = 2

  # Tear Production parsing
  col = 3
  if db[i][col] == "Reduced":
    X[i][col] = 1
  elif db[i][col] == "Normal":
    X[i][col] = 2
    
print(X)
      
#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
# Y =
for row in db:
  if row[4] == "Yes":
    Y.append(1)
  elif row[4] == "No":
    Y.append(2)
  else:
    Y.append(0)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()