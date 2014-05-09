''' builddt.py
#	Author			: Shivam Chaturvedi
#	Last Modified	: 01:49 AM, 10th September 2013
#	Purpose			: ID3 Implementation [Machine Learning](Decision classifier)
#	Copyright		: (C) 2013	'''

'''	NOTE: 	The builddt.py or the file that it is used in should call attribute.data file and instances.data file
			as command line parameters for the ID3 classifier to work properly '''

'''	The following functions are defined in the source code that follows:
	getEntropy(instances, targetAttribs)				:	calculates entropy for the given instances for the target Attributes
														:	returns a floating point number with 5 digits of precision
	
	getInfoGain(instances, attr, attrib, targetAttribs)	:	calculates the infoGain for the given instances
														:	attr is the total list of attributes
														:	attrib is the index of the selected attribute for which the infoGain is to be calculated
														:	targetAttribs are as usual
	
	Compute(instances, attrib, targetAttr)				:	Computes the decision tree in a recursive manner
														:	Final output is in a dictionary formatted tree
														:	The output tree can be compared with test data by the traverse() function
	
	traverse(decision, instance)						:	Takes the decision tree output from Compute function and checks the input instance to output the classified value
														:	Decision is the tree output by Compute
														:	instance is a list which is comouted by the makeinstance() function
	
	readAttributes(file)								:	Takes the input file and reads it for the Attribute list, except the target attributes
	
	readTargetAttributes(file)							:	Takes the input file and reads it for the Target Attribute list
	
	readInstances(file)									:	Takes the input file and reads it for the Instances
	
	'''
from math import log
import sys
import copy
#######FUNCTIONS#########
''' Generic function that will calculate the entropy of a given set of instances based on the target attribute value list supplied 
	@param 	: instances <list>
	@param	: targetAttribs <list>
	@return	: <float> precision:5		'''
def getEntropy(instances, targetAttribs):
	count = {}								#Will store the count of each target attribute's no. of occurences
	for target in targetAttribs:			#for each value of the target attrib,
		for instance in instances:			#in every instance
			if(instance[-1] == target):		#if the last value (i.e. the target) is the current
				if count.has_key(target):	#increment in count
					count[target] = count[target] + 1
				else:
					count[target] = 1
	entropy = 0
	totalinst = float(len(instances))		#total number of instances
	for key in count.keys():
		p = count[key] / totalinst			#calculate ratio of appearance or p
		entropy = entropy - p * log(p, 2)	#add to entropy as p * log2p
	return round(entropy, 5)

''' Generic function to calculate the InfoGain from a set of instances and target attribute with a certain attribute to calculate the Info gain against
	@param	: instances 	<list>
	@param	: attr			<list>
	@param	: attrib		<integer>
	@param	: targetAttribs	<list>
	@return	: <integer>	precision:5		'''	
	
def getInfoGain(instances, attr, attrib, targetAttribs):
	''' The formula is: entropy - Sigma((Sv / S) * entropy(Sv))
		We will calculate the Sigma part from this function
		Assuming that the instances are only those that are required to be used in this very Gain calculation
	'''
	# countVal = {}	#will hold the possible instances of a particular attribute in these instances
	# for target in targetAttribs:
		# for instance in instances:
			# if instance[-1] == target:
				# if countVal.has_key(attr[attrib][0]):
					# countVal[attr[attrib][0]] = countVal[attr[attrib][0]] + 1
				# else:
					# countVal[attr[attrib][0]] = 1
	# totalInst = countVal[attr[attrib][0]]	#total number of instances "S"
	totalInst = len(instances)
	#taking each value of the attribute
	#performing Sigma
	Sigma = 0								#to contain the whole Sigma values of Sv/S
	for eachValue in attr[attrib][1]:
		countTar = {}						#collect all the occurences of the target attributes in each value that the attribute can take up
		countTotal = 0.0					#count total number of instances of each value that the selected attribute can take
		eachSigma = 0						#count Sigma values of each of the 
		for eachInstance in instances:		#for each of the collected instances
			if eachInstance[attrib] == eachValue:			#if the value at the selected attribute matches the current selected value i.e. if the value is really v
				if countTar.has_key(eachInstance[-1]):		#increment the value in countTar if it present
					countTar[eachInstance[-1]] = countTar[eachInstance[-1]] + 1
				else:
					countTar[eachInstance[-1]] = 1
				countTotal = countTotal + 1.0				#keep a check on the total number of instances satisfying
				
		for i in countTar.keys():							#calculate p
			p = countTar[i] / countTotal
			eachSigma = eachSigma - p * log(p, 2)			#calculate eachSigma for this very value
		Sigma = Sigma + (countTotal / totalInst) * eachSigma	#add to the global Sigma value for the attribute
	return round(Sigma, 5)	#return the value to be used to subtract from Entropy to get InfoGain

''' Start the computation recursively and print the output tree in textual form  
	@param	: 	instances	<list>
	@param	: 	attrib		<list>
	@param	: 	targetAttr	<list>
	@return	: 	<string>	'''
def Compute(instances, attrib, targetAttr):
	entropy = getEntropy(instances, targetAttr)		#get entropy of this set
	#print "instances:" , len(instances), "attributes:" , len(attrib)
	#print "target attributes:" , len(targetAttr)
	if entropy == 0.00000:
		return [instances[0][-1]]
	if len(attrib) == 0:
		return {}
	maxInfoGain = 0									#initializations
	selectedAttribute = -1
	for attribute in range(len(attrib)):			#for every attribute, check whether they can be a candidate for root
		#print "checking for", attrib[attribute]
		
		attrGain = entropy - getInfoGain(instances, attrib, attribute, targetAttr)	#select every attribute from the attrib list and get individual values to subtract from entropy to get InfoGains
		#print "entropy: ", entropy, "gain for", attrib[attribute][0], "is", attrGain
		if attrGain != 0.00000 and attrGain > maxInfoGain:					#get maximum infogain
			maxInfoGain = attrGain
			selectedAttribute = attribute			#get the attribute number for the maxGain attribute
	if selectedAttribute == -1:
		return {0:instances[0][-1]}
	
	SAN = attrib[selectedAttribute][0] 				#SelectedAttributeName
	tree = {}
	#print "Selected attribute is:",SAN, str(selectedAttribute)
	thisAttrib = attrib.pop(selectedAttribute)
	for attribVal in thisAttrib[1]:
		#tree = tree + ("\t"*n) + "For " + SAN + " = " + attribVal + "\n"
		nextInstances = []
		for eachInstance in instances:
			if eachInstance[selectedAttribute] == attribVal:
				eachInstance.pop(selectedAttribute)
				nextInstances.append(copy.deepcopy(eachInstance))
				#instances.remove(eachInstance)
		if len(nextInstances) != 0:
			tree[str(SAN + "=" + attribVal)] = Compute(copy.deepcopy(nextInstances), copy.deepcopy(attrib), copy.deepcopy(targetAttr))
	return tree

''' Converts the comma separated user test data input into the format that can be checked by the classifier 
	@param: 	user	<string>
	@return: 	<string>	'''
def makeinstance(user):
	#fetch the attribute file again as the attr variable gets modified during making of the decision tree

	f = open(sys.argv[1])						#Load Attribute data file
		
	attr = []											#attr will contain all the attributes that there can be and their possible values
	targetAttr = f.readline().split(",")				#read the target attributes' values
	targetAttr[-1] = targetAttr[-1].replace("\n", "")	#remove the trailing \n
	for eachline in f:									#read the rest of the file
		eachAttrib = eachline.split(":")				#get the attribute name, list of values
		attrVal = eachAttrib[1].split(",")				#get all the other possible values
		attrVal[-1] = attrVal[-1].replace("\n", "")
		a = [(eachAttrib[0]), attrVal]					#first item will be the name of attrib, 2nd a list of values
		attr.append(a)
	
	user = user.split(",")
	instance = []
	for i in range(len(attr)):		#construct a concatenated list of attribute=value pair for the input
		instance.append(attr[i][0] + "=" + user[i])
	return instance

'''	traverse function that traverses the decision tree built on the supplied input 
	@param:	decision	<dictionary>
	@param: instance	<list>
	@return: <string>	'''
def traverse(decision, instance, debug = 0):
	for key in instance:					#For every key in the input to be classified
		# if debug == 1:
			# print "\n\t\tSearching for the key", key, "in", decision
		if decision.has_key(key):			#if the current root has decision, then
			if type(decision[key]) is list:		#if there is a single node i.e. leaf, then that is the decision
				# if debug == 1:
					# print "\n\t\tTrying for", decision[key]
					# print "check"
				# if type(decision[key][decision[key].keys()[0]]) is dict:
				# print "returned decision"
				return str(decision[key][0])
				# else:
					# print (type(decision[decision.keys()[0]]) is dict)
			else:							#else recurse in the subtree of the decision tree
				nextdecision = decision[key]	#recurse into the tree
				instance.remove(key)		#remove the attribute that made the decision
				# if debug == 1:
					# print "\n\tinstance", instance, "\n\tdecision", decision
				return traverse(nextdecision, instance, debug)
	
	return "-1"

'''the decision tree is formed as: {attribute1: {attribute1.1: {}}, decision2:{}}... 
	attributes.data should contain data like this:
	1st line: the target attribute's values separated by comma
	following N lines: the other attributes followed by a : and their possible values
'''
''' Reads in Attributes from the file, typically the first parameter provided
	@param	:	file	<string>
	@return	:	<list>	'''
def readAttributes(file):
	f = open(file)						#Load Attribute data file

	attr = []											#attr will contain all the attributes that there can be and their possible values
	targetAttr = f.readline().split(",")				#read the target attributes' values
	targetAttr[-1] = targetAttr[-1].replace("\n", "")	#remove the trailing \n
	for eachline in f:									#read the rest of the file
		eachAttrib = eachline.split(":")				#get the attribute name, list of values
		attrVal = eachAttrib[1].split(",")				#get all the other possible values
		attrVal[-1] = attrVal[-1].replace("\n", "")
		a = [(eachAttrib[0]), attrVal]					#first item will be the name of attrib, 2nd a list of values
		attr.append(a)									#store in the attr list as index ->attr -> value triplets
	return attr

''' reads in Target Attributes from the file, typically the first parameter provided
	@param	:	file	<string>
	@return	:	<list>	'''
def readTargetAttributes(file):
	f = open(file)						#Load Attribute data file

	attr = []											#attr will contain all the attributes that there can be and their possible values
	targetAttr = f.readline().split(",")				#read the target attributes' values
	targetAttr[-1] = targetAttr[-1].replace("\n", "")	#remove the trailing \n
	return targetAttr

'''	The instances.data should contain the N instances to be used as training data set with each line containing one instance, with all the attributes in order as in attr.data and the last attribute should be the target attribute	'''
''' reads in Instances from the file, typically the second parameter provided
	@param	:	file	<string>
	@return	:	<list>	'''
def readInstances(file):
	f = open(file)						#Load instances data file
		
	f.seek(0)							#reach the beginning of file (if not so already)
	inst = []							#the main list that will contain each of the instances
	for thisInstance in f:
		a=thisInstance.split(",")		#split the values in each instance...
		a[-1] = a[-1].replace("\n", "")	#correct the \n
		inst.append(a)					#add to the list of instances
	return inst

#######################'''START'''############################
'''
This script should be loaded with the following parameters, which if not provided, will search for default files:
1. @param[1]	: attributes.data
2. @param[2]	: instances.data
'''
print "####\\\\ID3 Decision Classifier////####\n"

################'''START MAIN COMPUTATION'''##################

# decision = Compute(inst, attr, targetAttr)
# attributes = readAttributes(sys.argv[1])
# targetAttributes = readTargetAttributes(sys.argv[1])
# instances = readInstances(sys.argv[2])