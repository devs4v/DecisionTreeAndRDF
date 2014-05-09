'''	RDForestCV.py
#	Author			: Shivam Chaturvedi
#	Last Modified	: 06:25 PM, 10th September 2013
#	Purpose			: Perform K-fold cross validation on RDF[Machine Learning](Checking accuracy of classified decisions using k-fold validation)
#	Copyright		: (C) 2013
'''

from random import random
import sys
from rdforest import *	#importing the RDForest classifier file (includes the ID3 classifier for working)


print "Performing K-fold cross validation on input data"
#print "##############''''Now taking random decision tree''''############"
#	'''K-Fold Parameters
k = 5				#no. of folds
print "Performing cross validation for", k, "folds"
a = raw_input("Enter a value if you want a different 'k' value for the number of folds.\nElse press Enter\tAnswer: ")
if a != "":
	k = int(a)
	
numTrees = 10
fractionOfInstances = 0.6

a = raw_input("Building a Random Decision Forest with " + str(numTrees) + " Trees.\nHit Enter to Continue, or enter the number of Trees required and press enter\nAnswer:")
if a != "":
	numTrees = int(a)

a = raw_input("Taking " + str(numTrees) + " fraction of Instances each time.\nHit Enter to Continue, or enter another fraction and press enter\nAnswer:")
if a != "":
	fractionOfInstances = float(a)

attributes = readAttributes(sys.argv[1])
targetAttributes = readTargetAttributes(sys.argv[1])
instances = readInstances(sys.argv[2])

############# STARTING CROSS VALIDATION ########
TruePositive = 0
TotalCases = 0


attributes = readAttributes(sys.argv[1])
targetAttributes = readTargetAttributes(sys.argv[1])
instances = readInstances(sys.argv[2])

x = len(instances)		#the inst variable keeps all the instances that are read
fold = x/k				#length of one fold
for j in range(k):
	foldTruePositive = 0
	foldCases = 0
	dtrees = []
	start = j*fold			#calculating the start and end of the fold that will be used for the test data
	end = ((j+1)*fold)
	print "\n#######Iteration :", (j+1)
	trainingInstances = []					#variables to separate out the training and test instances on each iteration
	testInstances = []
	for i in range(x):
		if (i < start or i >= end):
			trainingInstances.append(copy.deepcopy(instances[i]))
		else:
			testInstances.append(copy.deepcopy(instances[i]))
	
	dtrees = makeRDForest(trainingInstances, attributes, targetAttributes, numTrees, fractionOfInstances)	#build the forest
	#print dtrees
	for testInstance in testInstances:
		
		actualAnswer = testInstance[-1]
	
		testInstance = makeinstance(disassembleInstance(testInstance))
		
		#if (j+1) != 5:
		answer = decideOnInstance(copy.deepcopy(dtrees), copy.deepcopy(testInstance))
		# else:
			# answer = traverse(dtree, instance)
		
		if actualAnswer == answer:
			foldTruePositive = foldTruePositive + 1
		foldCases = foldCases + 1
		
	print "%age of TruePositives: " + str(round(((foldTruePositive/float(foldCases)) * 100), 2)) + "%"
		
	TruePositive = TruePositive + foldTruePositive
	TotalCases = TotalCases + foldCases
	
print "The classification accuracy in the " + str(k) + " folds was: " +  str(round(((TruePositive/float(TotalCases)) * 100), 2)) + "%"