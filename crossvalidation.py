'''	crossvalidation.py
#	Author			: Shivam Chaturvedi
#	Last Modified	: 10:52 PM, 8th September 2013
#	Purpose			: Perform cross validation [Machine Learning](Checking accuracy of classified decisions using k-fold validation)
#	Copyright		: (C) 2013
'''

from random import random
import sys
from builddt import *	#importing the ID3 classifier file


	
#print "##############''''Now taking random decision tree''''############"
k = 5				#no. of folds
print "Performing cross validation for", k, "folds"
a = raw_input("Enter a value if you want a different 'k' value for the number of folds.\nElse press Enter\tAnswer: ")
if a != "":
	k = int(a)

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
	
	dtree = Compute(copy.deepcopy(trainingInstances), copy.deepcopy(attributes), copy.deepcopy(targetAttributes))
	#print dtree
	for instance in testInstances:
		
		instanceStr = ""
		for i in instance:
			instanceStr = instanceStr + str(i)
			if i != instance[-1]:
				instanceStr = instanceStr + ","
				
		instance = makeinstance(instanceStr)
		if (j+1) != 5:
			answer = traverse(dtree, instance)
		else:
			answer = traverse(dtree, instance, 1)
		
		if instanceStr.split(",")[-1] == answer:
			foldTruePositive = foldTruePositive + 1
		foldCases = foldCases + 1
		
	print "%age of TruePositives: " + str(round(((foldTruePositive/float(foldCases)) * 100), 2)) + "%"
		
	TruePositive = TruePositive + foldTruePositive
	TotalCases = TotalCases + foldCases
print "The classification accuracy in the " + str(k) + " folds was: " +  str(round(((TruePositive/float(TotalCases)) * 100), 2)) + "%"