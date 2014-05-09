''' rdforest.py
#	Author			: Shivam Chaturvedi
#	Created			: 10:25 PM, 9th September 2013
#	Last Modified	: 03:39 AM, 10th September 2013
#	Purpose			: [Machine Learning] Random Decision Forest Implementation (using ID3 classifier [by Shivam Chaturvedi])
#	Copyright		: (C) 2013
'''

from builddt import *
from random import random
from random import shuffle

print "####\\\\Random Decision Forest Classifier////####"

def disassembleInstance(instance):
	s = ""
	for i in instance:
		s = s + i
		if s != instance[-1]:
			s = s + ","
		
	return s

def makeRDForest(instances, attributes, targetAttributes, numTrees = 10, fraction = 0.6):
	fold = int(len(instances) * fraction)
	# print fold
	dtrees = []
	for aTree in range(numTrees):
		#every iteration
		randomInstances = []
		randomCollection = []
		while(len(randomCollection) < fold):
			num = int(random()*len(instances))
			if num not in randomCollection:
				randomCollection.append(num)
		
		for i in randomCollection:
			randomInstances.append(copy.deepcopy(instances[i]))
			
		dtrees.append(Compute(copy.deepcopy(randomInstances), copy.deepcopy(attributes), copy.deepcopy(targetAttributes)))
		#print str(aTree) + "=====" + str(dtrees)
	return dtrees

def decideOnInstance(dtrees, testInstance):	
	answers = {}
	for tree in dtrees:
		#print "traversing", tree
		answer = traverse(copy.deepcopy(tree), copy.deepcopy(testInstance),1)
		#print answer
		if answer != "-1":		#To remove those instances when a particular decision tree could not find anything
			if answer in answers.keys():
				answers[answer] = answers[answer] + 1
			else:
				answers[answer] = 1

	if	len(answers.values()) != 0:
		decision = max(answers.values())
		for answer in answers.keys():
			if answers[answer] == decision:
				decision = answer
				return decision
	else:
		return "-1"
#We will need multiple sets of random instances, that we will use to make multiple decision trees
#Then we will take an input and then feed it to all the decision trees one by one
#Whatever decision the majority takes will be the final decision and then the same checking procedure can be applied

'''
The following commented out code is the sample code to run the RDF classifier on a random instance from among the instance
Almost models the leave one out cross validation strategy
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
print "The classification is:",d[decideOnInstance(dtrees, testInstance) == actualAnswer]
'''