'''	crossvalidation.py
#	Author			: Shivam Chaturvedi
#	Last Modified	: 06:25 PM, 10th September 2013
#	Purpose			: Perform random cross validation [Machine Learning](Checking accuracy of classified decisions using k-fold validation)
#	Copyright		: (C) 2013
'''

from random import random
import sys
from rdforest import *	#importing the RDForest classifier file (includes the ID3 classifier for working)

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

dtrees = makeRDForest(instances, attributes, targetAttributes, numTrees, fractionOfInstances)
	
#Making a test instance
testInstance = instances[int(random()*len(instances))]
actualAnswer = testInstance[-1]
	
testInstance = makeinstance(disassembleInstance(testInstance))

d = {1: "Correct", 0: "Incorrect"}
print "The classification is:",d[decideOnInstance(copy.deepcopy(dtrees), testInstance) == actualAnswer]
