# Code provided by Tomojit Ghosh(tomojit.ghosh@colostate.edu) and Michael Kirby (Kirby@math.colostate.edu)
#
# Copyright (c) 2020 Tomojit Ghosh  and Michael Kirby

# Permission is granted, free of charge, to everyone to copy, use, modify, or distribute this
# program and accompanying programs and documents for any purpose, provided
# this copyright notice is retained and prominently displayed, along with
# a note saying that the original programs are . 
# 
# The software is provided "as is", without any warranty, express or
# implied.  As the programs were written for research purposes only, they have
# not been tested to the degree that would be advisable in any important
# application.  All use of these programs is entirely at the user's own risk.

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp
import random
from operator import itemgetter
import pickle

def getApplicationData(dataSetName,nSamplePerClass=0):

	if dataSetName.upper()=='MNIST':
		trFile,trLabelFile = './DataFile/MNIST/MNISTTrainData_Org.p','./DataFile/MNIST/MNISTTrainLabel_Org.p'
		trData,trLabels = pickle.load(open(trFile,'rb')),pickle.load(open(trLabelFile,'rb'))

		lTrData = np.hstack((trData,trLabels))
		trData,trLabels=makeStratifiedSubset(lTrData,nSamplePerClass)
	    
		testFile,testLabelFile = './DataFile/MNIST/MNISTTestData_Org.p','./DataFile/MNIST/MNISTTestLabel_Org.p'
		tstData,tstLabels = pickle.load(open(testFile,'rb')),pickle.load(open(testLabelFile,'rb'))
		lTstData = np.hstack((tstData,tstLabels))
		tstData,tstLabels=makeStratifiedSubset(lTstData,nSamplePerClass)

	else: #for USPS data
		labeledDataFile = './DataFile/USPS/USPS.p'
		labeledData = pickle.load(open(labeledDataFile,'rb'))
		trData,trLabels = labeledData[:,:-1],labeledData[:,-1].reshape(-1,1)
		tstData,tstLabels=[],[]

	return trData,trLabels,tstData,tstLabels

def makeStratifiedSubset(labeledData,noSamplePerClass=0):
	#This function will randomly pick 'noSamplePerClass' samples from each class to create a subset.
	#If noSamplePerClass==0, then all the data will be returned.
	data = []
	label = []
	classList = np.unique(labeledData[:,-1])
	for c in classList:
		indices = np.where(labeledData[:,-1]==c)[0]
		if noSamplePerClass==0:#take all the samples of this class
			data.append(labeledData[indices,:-1])
			label.append(labeledData[indices,-1])
		else:
			data.append(labeledData[indices[:noSamplePerClass],:-1])
			label.append(labeledData[indices[:noSamplePerClass],-1])
	return np.vstack((data)),np.hstack((label)).reshape(-1,1)

def splitData_n(data,labels,nTrData):
	# This method will split the dataset into training and test set.
	# The labeled_data will be shuffled and then first nTrData will be put in training set
	# and the rest will be put together in test set
	train_set =[]
	test_set =[]
	no_data = len(data)
	dataDim = np.shape(data)[1]
	indices = np.arange(no_data)
	np.random.shuffle(indices)
	trData,trLabels = data[indices[:nTrData],:],labels[indices[:nTrData]].reshape(-1,1)
	lTrData = np.hstack((trData,trLabels))
	sortedTrData = np.array(sorted(lTrData, key=itemgetter(dataDim)))
	trData,trLabels = sortedTrData[:,:-1],sortedTrData[:,-1]
	tstData,tstLabels = data[indices[nTrData:],:],labels[indices[nTrData:]].reshape(-1,1)
	lTstData = np.hstack((tstData,tstLabels))
	sortedTstData = np.array(sorted(lTstData, key=itemgetter(dataDim)))
	tstData,tstLabels = sortedTstData[:,:-1],sortedTstData[:,-1]
	return trData,trLabels.reshape(-1,1),tstData,tstLabels.reshape(-1,1)

def standardizeData(data,mu=[],std=[]):
	#data: a m x n matrix where m is the no of observations and n is no of features
	#if any(mu) == None and any(std) == None:
	if not(len(mu) and len(std)):
		#pdb.set_trace()
		std = np.std(data,axis=0)
		mu = np.mean(data,axis=0)
		std[np.where(std==0)[0]] = 1.0 #This is for the constant features.
		standardizeData = (data - mu)/std
		return mu,std,standardizeData
	else:
		standardizeData = (data - mu)/std
		return standardizeData
		
def unStandardizeData(data,mu,std):
	return std * data + mu

def makeAnnotationMNIST(labels):
	annotData = []
	a0 = [1,'tomato','+',25,'0']
	a1 = [2,'lawngreen','+',25,'1']
	a2 = [3,'gold','+',25,'2']
	a3 = [4,'darkgreen','+',25,'3']
	a4 = [5,'m','+',25,'4']
	a5 = [6,'mediumspringgreen','+',25,'5']
	a6 = [7,'k','+',25,'6']
	a7 = [8,'royalblue','+',25,'7']
	a8 = [9,'brown','+',25,'8']
	a9 = [10,'deepskyblue','+',25,'9']
	#for c in np.unique(labels):
	for c in range(len(labels)):		
		if labels[c] == 0:			
			annotData.append(a0)
		elif labels[c] == 1:			
			annotData.append(a1)
		elif labels[c] == 2:
			annotData.append(a2)
		elif labels[c] == 3:
			annotData.append(a3)
		elif labels[c] == 4:
			annotData.append(a4)
		elif labels[c] == 5:
			annotData.append(a5)
		elif labels[c] == 6:
			annotData.append(a6)
		elif labels[c] == 7:
			annotData.append(a7)
		elif labels[c] == 8:
			annotData.append(a8)
		elif labels[c] == 9:
			annotData.append(a9)
	return np.vstack((annotData))

def makeAnnotationUSPS(labels):
	annotData = []
	a0 = [1,'tomato','+',25,'0']
	a1 = [2,'lawngreen','+',25,'1']
	a2 = [3,'gold','+',25,'2']
	a3 = [4,'darkgreen','+',25,'3']
	a4 = [5,'m','+',25,'4']
	a5 = [6,'brown','+',25,'5']
	a6 = [7,'c','+',25,'6']
	a7 = [8,'royalblue','+',25,'7']
	a8 = [9,'deepskyblue','+',25,'8']
	a9 = [10,'k','+',25,'9']

	for c in range(len(labels)):		
		if labels[c] == 0:			
			annotData.append(a0)
		elif labels[c] == 1:			
			annotData.append(a1)
		elif labels[c] == 2:
			annotData.append(a2)
		elif labels[c] == 3:
			annotData.append(a3)
		elif labels[c] == 4:
			annotData.append(a4)
		elif labels[c] == 5:
			annotData.append(a5)
		elif labels[c] == 6:
			annotData.append(a6)
		elif labels[c] == 7:
			annotData.append(a7)
		elif labels[c] == 8:
			annotData.append(a8)
		elif labels[c] == 9:
			annotData.append(a9)

	return np.vstack((annotData))

def display2DData(trData,trAnnotation,fig_l):
	'''
	#This function can be used to do the scatter plot
	#annotation_data has the following fields:
	#field1: Group number starts from 1
	#field2: color code
	#field3: Shape
	#field4: Size
	#field5: Label
	'''
	fig1 = plt.figure(1)
	ax1 = fig1.add_subplot(111)
	#Now iterate for each data and plot
	group = []
	for i in range(len(trAnnotation)):
		if(int(trAnnotation[i,0]) not in group):
			if trAnnotation[i,2] == '+':
				ax1.scatter(trData[i,0],trData[i,1],s=trAnnotation[i,3].astype(int), edgecolors=str(trAnnotation[i,1]), facecolors=str(trAnnotation[i,1]),marker=trAnnotation[i,2],label = trAnnotation[i,4])
			else:
				ax1.scatter(trData[i,0],trData[i,1],s=trAnnotation[i,3].astype(int), edgecolors=str(trAnnotation[i,1]), facecolors='None',marker=trAnnotation[i,2],label = trAnnotation[i,4])
				
			group.extend([int(trAnnotation[i,0])])
		else:
			if trAnnotation[i,2] == '+':
				ax1.scatter(trData[i,0],trData[i,1],s=trAnnotation[i,3].astype(int), edgecolors=str(trAnnotation[i,1]), facecolors=str(trAnnotation[i,1]),marker=trAnnotation[i,2])
			else:
				ax1.scatter(trData[i,0],trData[i,1],s=trAnnotation[i,3].astype(int), edgecolors=str(trAnnotation[i,1]), facecolors='None',marker=trAnnotation[i,2])
			
	plt.legend(loc='upper right',fontsize=10)
	ax1.set_yticklabels([])
	ax1.set_xticklabels([])
	fig1.suptitle(fig_l,fontsize=15)
	ax1.axis('off')
	plt.show()

def display2DDataTrTst(trData,trCentroids,trAnnotation,tstData,tstAnnotation,dataSetName):
	
	# data: a dictionary. The keys are the model names. In each model there are training and test data with labels
	fig, ax = plt.subplots(1,2)
	
	# first plot the training data
	vorTr = sp.Voronoi(trCentroids)
	sp.voronoi_plot_2d(vorTr, show_vertices=False, line_colors='orange',line_width=1, line_alpha=0.6, point_size=5,ax=ax[0])
	group = []
	#pdb.set_trace()
	for i in range(len(trAnnotation)):	   
		if(int(trAnnotation[i,0]) not in group):
			if trAnnotation[i,2] == '+':
				ax[0].scatter(trData[i,0],trData[i,1],s=trAnnotation[i,3].astype(int), edgecolors=str(trAnnotation[i,1]), facecolors=str(trAnnotation[i,1]),marker=trAnnotation[i,2],label = trAnnotation[i,4])
			else:
				ax[0].scatter(trData[i,0],trData[i,1],s=trAnnotation[i,3].astype(int), edgecolors=str(trAnnotation[i,1]), facecolors='None',marker=trAnnotation[i,2],label = trAnnotation[i,4])
			group.extend([int(trAnnotation[i,0])])
		else:
			if trAnnotation[i,2] == '+':
				ax[0].scatter(trData[i,0],trData[i,1],s=trAnnotation[i,3].astype(int), edgecolors=str(trAnnotation[i,1]), facecolors=str(trAnnotation[i,1]),marker=trAnnotation[i,2])
			else:
				ax[0].scatter(trData[i,0],trData[i,1],s=trAnnotation[i,3].astype(int), edgecolors=str(trAnnotation[i,1]), facecolors='None',marker=trAnnotation[i,2])
		
	ax[0].title.set_text('Training data')
	ax[0].set_yticklabels([])
	ax[0].set_xticklabels([])

	# now plot the test data
	vorTst = sp.Voronoi(trCentroids)
	sp.voronoi_plot_2d(vorTr, show_vertices=False, line_colors='orange',line_width=1, line_alpha=0.6, point_size=5,ax=ax[1])
	group = []
	for i in range(len(tstAnnotation)):	   
		if(int(tstAnnotation[i,0]) not in group):
			if tstAnnotation[i,2] == '+':
				ax[1].scatter(tstData[i,0],tstData[i,1],s=tstAnnotation[i,3].astype(int), edgecolors=str(tstAnnotation[i,1]), facecolors=str(tstAnnotation[i,1]),marker=tstAnnotation[i,2],label = tstAnnotation[i,4])
			else:
				ax[1].scatter(tstData[i,0],tstData[i,1],s=tstAnnotation[i,3].astype(int), edgecolors=str(tstAnnotation[i,1]), facecolors='None',marker=tstAnnotation[i,2],label = tstAnnotation[i,4])
			group.extend([int(tstAnnotation[i,0])])
		else:
			if tstAnnotation[i,2] == '+':
				ax[1].scatter(tstData[i,0],tstData[i,1],s=tstAnnotation[i,3].astype(int), edgecolors=str(tstAnnotation[i,1]), facecolors=str(tstAnnotation[i,1]),marker=tstAnnotation[i,2])
			else:
				ax[1].scatter(tstData[i,0],tstData[i,1],s=tstAnnotation[i,3].astype(int), edgecolors=str(tstAnnotation[i,1]), facecolors='None',marker=tstAnnotation[i,2])

	ax[1].title.set_text('Test data')
	ax[1].set_yticklabels([])
	ax[1].set_xticklabels([])

	if dataSetName.upper() in ['MNIST','USPS']:
		plt.legend(loc='upper center', bbox_to_anchor=(0.0005, -0.0025),ncol=10, fancybox=False, shadow=False,fontsize=15)
		#plt.legend(bbox_to_anchor=(0.00025, 1.65),ncol=1, fancybox=False, shadow=False,fontsize=15)
	else:
		plt.legend(loc='upper right',fontsize=15)
	plt.plot()
	plt.show()

def calcCentroid(data,label):
	centroids=[]
	centroidLabels=np.unique(label)
	for i in range(len(centroidLabels)):
		tmpData=data[np.where(centroidLabels[i]==label)[0],:]
		centroids.append(np.mean(tmpData,axis=0))
	centroids=np.vstack((centroids))
	return centroids
	
def createOutputAsCentroids(data,label):
	centroidLabels=np.unique(label)
	outputData=np.zeros([np.shape(data)[0],np.shape(data)[1]])
	for i in range(len(centroidLabels)):
		indices=np.where(centroidLabels[i]==label)[0]
		tmpData=data[indices,:]
		centroid=np.mean(tmpData,axis=0)
		outputData[indices,]=centroid
	return outputData

def returnBottleneckArc(dataSetName):
	if dataSetName.upper()=='USPS':
		return [2000,1000,500,2]
	elif dataSetName.upper()=='MNIST':
		return [1000,500,125,2]
	
def returnActFunc(dataSetName):
	if dataSetName.upper()=='USPS':
		return ['relu','relu','relu','linear']
	elif dataSetName.upper()=='MNIST':
		return ['tanh','tanh','tanh','linear']
