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
import sys
from utilityScript import *
from SupervisedCentroidencodeVisualizerPyTorch import SCEVisualizer

def createModel(dataSetName):

	# load data
	if dataSetName == 'MNIST':
		trData,trLabels,tstData,tstLabels = getApplicationData(dataSetName)
		
		# only use 1000 samples per class(total 10,000 samples) from training data for visualization
		nSamplePerClass = 1000
		trDataViz,trLabelsViz,_,_ = getApplicationData(dataSetName,nSamplePerClass)
		tstDataViz,tstLabelsViz = tstData,tstLabels
		annotDataTr = makeAnnotationMNIST(trLabelsViz)
		annotDataTst = makeAnnotationMNIST(tstLabelsViz)
	
	else: # for USPS
		orgData,orgLabels,tstData,tstLabels = getApplicationData(dataSetName)
		nTrData = 8000
		trData,trLabels,tstData,tstLabels = splitData_n(orgData,orgLabels,nTrData)
		annotDataTr = makeAnnotationUSPS(trLabels)
		annotDataTst = makeAnnotationUSPS(tstLabels)

	# hyper-parameters for Adam optimizer
	if dataSetName == 'MNIST':
		num_epochs_pre = 25
		num_epochs_post = 200
		miniBatch_size = 512
		learning_rate = 0.0008
	else: #for USPS
		num_epochs_pre = 25
		num_epochs_post = 50
		miniBatch_size = 64
		learning_rate = 0.001

	# parameters for the CE network
	dict2 = {}
	dict2={}
	dict2['inputDim']=np.shape(trData)[1]
	dict2['hL']=returnBottleneckArc(dataSetName)
	dict2['hActFunc']= returnActFunc(dataSetName)
	dict2['oActFunc']='linear'
	dict2['errorFunc']='MSE'
	dict2['l2Penalty']=0.00002

	standardizeFlag = True
	preTrFlag = True

	# build a model
	model = SCEVisualizer(dict2)
	print('Training centroid-encoder to build a model.')
	
	# train the model
	model.fit(trData,trLabels,
			learningRate=learning_rate,
			miniBatchSize=miniBatch_size,
			numEpochsPreTrn=num_epochs_pre,
			numEpochsPostTrn=num_epochs_post,
			standardizeFlag=standardizeFlag,
			preTraining=preTrFlag)
	
	# reduce dimension of training and test data
	if dataSetName == 'MNIST':
		pDataTr = model.predict(trDataViz)[len(dict2['hL'])].to('cpu').numpy()
		trCentroids = calcCentroid(pDataTr,trLabelsViz)
		pDataTst = model.predict(tstDataViz)[len(dict2['hL'])].to('cpu').numpy()
	else:
		pDataTr = model.predict(trData)[len(dict2['hL'])].to('cpu').numpy()
		trCentroids = calcCentroid(pDataTr,trLabels)
		pDataTst = model.predict(tstData)[len(dict2['hL'])].to('cpu').numpy()
	
	# now visualize the training and test data using voronoi cells
	display2DDataTrTst(pDataTr,trCentroids,annotDataTr,pDataTst,annotDataTst,dataSetName)

if __name__== "__main__":
	if len(sys.argv)-1 == 0:
		print("Missing dataset name, visualizing default dataset: USPS.")
		dataSet = 'USPS'
	else:
		dataSet = sys.argv[1].upper()
		print("Visualizing dataset:",dataSet)
	createModel(dataSet)
