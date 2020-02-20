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

from copy import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from utilityScript import createOutputAsCentroids,standardizeData

class SCEVisualizer(nn.Module):
	def __init__(self, netConfig={}):
		super(SCEVisualizer, self).__init__()
		if len(netConfig.keys())!=0:		
			self.inputDim,self.outputDim=netConfig['inputDim'],netConfig['inputDim']
			self.hLayer,self.hLayerPost=copy(netConfig['hL']),copy(netConfig['hL'])
			for i in range(len(self.hLayerPost)-1,0,-1):
				self.hLayerPost.extend([self.hLayerPost[i-1]])

			self.l1Penalty,self.l2Penalty=0.0,0.0
			self.oActFunc,self.errorFunc='linear','MSE'

			#pdb.set_trace()
			if 'l1Penalty' in netConfig.keys(): self.l1Penalty=netConfig['l1Penalty']
			if 'l2Penalty' in netConfig.keys(): self.l2Penalty=netConfig['l2Penalty']			
			if 'errorFunc' in netConfig.keys(): self.errorFunc=netConfig['errorFunc']
			if 'oActFunc' in netConfig.keys(): self.oActFunc=netConfig['oActFunc']

			#pdb.set_trace()
			self.hActFunc,self.hActFuncPost=copy(netConfig['hActFunc']),copy(netConfig['hActFunc'])
			for i in range(len(self.hActFuncPost)-1,0,-1):
				self.hActFuncPost.extend([self.hActFuncPost[i-1]])
		else:#for default set up
			self.hLayer=[2]
			self.oActFunc,self.errorFunc='linear','MSE'
			self.hActFunc,self.hActFuncPost='tanh','tanh'

		self.device = None

		#internal variables
		self.epochError=[]
		self.trMu=[]
		self.trSd=[]
		self.tmpPreHActFunc=[]

	def initNet(self,input_size,hidden_layer):
		self.hidden=nn.ModuleList()
		# Hidden layers
		if len(hidden_layer)==1:
			self.hidden.append(nn.Linear(input_size,hidden_layer[0]))
		elif(len(hidden_layer)>1):
			for i in range(len(hidden_layer)-1):
				if i==0:
					self.hidden.append(nn.Linear(input_size, hidden_layer[i]))
					self.hidden.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
				else:
					self.hidden.append(nn.Linear(hidden_layer[i],hidden_layer[i+1]))
		self.reset_parameters(hidden_layer)
		# Output layer
		self.out = nn.Linear(hidden_layer[-1], input_size)

	def reset_parameters(self,hidden_layer):
		#pdb.set_trace()
		tmpActFunc = self.hActFunc[:int(np.ceil(len(hidden_layer)/2))]
		for i in range(len(tmpActFunc)-1,0,-1):
			tmpActFunc.extend([tmpActFunc[i-1]])
		hL = 0
		
		while True:
			#pdb.set_trace()
			if tmpActFunc[hL].upper() in ['SIGMOID','TANH']:
				#pdb.set_trace()
				torch.nn.init.xavier_uniform_(self.hidden[hL].weight)
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
				#continue
			elif tmpActFunc[hL].upper() == 'RELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			elif tmpActFunc[hL].upper() == 'LRELU':
				torch.nn.init.kaiming_uniform_(self.hidden[hL].weight, mode='fan_in', nonlinearity='leaky_relu')
				if self.hidden[hL].bias is not None:
					torch.nn.init.zeros_(self.hidden[hL].bias)
			if hL == len(hidden_layer)-1:
				break
			hL += 1

	def forwardPost(self, x):
		# Feedforward
		for l in range(len(self.hidden)):
			if self.hActFuncPost[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.hActFuncPost[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
			else:#default is linear				
				x = self.hidden[l](x)

		if self.oActFunc.upper()=='SIGMOID':
			return torch.sigmoid(self.out(x))
		else:
			return self.out(x)

	def forwardPre(self, x):
		#pdb.set_trace()
		# Feedforward
		for l in range(len(self.hidden)):
			if self.tmpPreHActFunc[l].upper()=='SIGMOID':
				x = torch.sigmoid(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='TANH':
				x = torch.tanh(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='RELU':
				x = torch.relu(self.hidden[l](x))
			elif self.tmpPreHActFunc[l].upper()=='LRELU':
				x = F.leaky_relu(self.hidden[l](x),inplace=False)
			else:#default is linear
				x = self.hidden[l](x)
		if self.oActFunc.upper()=='SIGMOID':
			return torch.sigmoid(self.out(x))
		else:
			return self.out(x)

	def setHiddenWeight(self,W,b):
		for i in range(len(self.hidden)):
			self.hidden[i].bias.data=b[i].float()
			self.hidden[i].weight.data=W[i].float()

	def setOutputWeight(self,W,b):
		self.out.bias.data=b.float()
		self.out.weight.data=W.float()

	def returnTransformedData(self,x):
		fOut=[x]
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			for layer in self.hidden:
				fOut.append(self.hiddenActivation(layer(fOut[-1])))
			if self.output_activation.upper()=='SIGMOID':
				fOut.append(torch.sigmoid(self.out(fOut[-1])))
			else:
				fOut.append(self.out(fOut[-1]))
		return fOut[1:]#Ignore the original input

	def preTrain(self,dataLoader,learningRate,batchSize,numEpochs,verbose):

		# set device
		device = self.device
		
		#loop to do layer-wise pre-training
		for d in range(len(self.hLayer)):
			
			#set the hidden layer structure for a bottleneck architecture
			hidden_layer=self.hLayer[:d+1]
			self.tmpPreHActFunc=self.hActFunc[:d+1]
			for i in range(len(hidden_layer)-1,0,-1):
				hidden_layer.extend([hidden_layer[i-1]])
				self.tmpPreHActFunc.extend([self.tmpPreHActFunc[i-1]])

			if verbose:
				if d==0:
					print('Pre-training layer [',self.inputDim,'-->',hidden_layer[0],'-->',self.inputDim,']')
				else:
					index=int(len(hidden_layer)/2)
					print('Pre-training layer [',hidden_layer[index-1],'-->',hidden_layer[index],'-->',hidden_layer[index+1],']')			

			#initialize the network weight and bias
			self.initNet(self.inputDim,hidden_layer)

			#freeze pretrained layers
			if d>0:
				j=0#index for preW and preB
				for l in range(len(hidden_layer)):
					if (l==d) or (l==(d+1)):
						continue
					else:
						self.hidden[l].weight=preW[j]
						self.hidden[l].weight.requires_grad=False
						self.hidden[l].bias=preB[j]
						self.hidden[l].bias.requires_grad=False
						j+=1
				self.out.weight=preW[-1]
				self.out.weight.requires_grad=False
				self.out.bias=preB[-1]
				self.out.bias.requires_grad=False

			# set loss function
			if self.errorFunc.upper() == 'CE':
				criterion = nn.CrossEntropyLoss()
			elif self.errorFunc.upper() == 'BCE':
				criterion = nn.BCELoss()
			elif self.errorFunc.upper() == 'MSE':
				criterion = nn.MSELoss()

			# set optimization function
			optimizer = torch.optim.Adam(self.parameters(),lr=learningRate,amsgrad=True)
			
			# Load the model to device
			self.to(device)

			# Start training
			for epoch in range(numEpochs):
				error=[]
				for i, (trInput, trOutput) in enumerate(dataLoader):  
					# Move tensors to the configured device
					trInput = trInput.to(device)
					trOutput = trOutput.to(device)

					# Forward pass
					outputs = self.forwardPre(trInput)
					loss = criterion(outputs, trOutput)
					
					# Check for regularization
					if self.l1Penalty != 0 or self.l2Penalty != 0:
						l1RegLoss,l2RegLoss = torch.tensor([0.0],requires_grad=True).to(device), torch.tensor([0.0],requires_grad=True).to(device)
						if self.l1Penalty != 0 and self.l2Penalty == 0:
							for W in self.parameters():
								l1RegLoss += W.norm(1)
							loss = loss + self.l1Penalty * l1RegLoss
						elif self.l1Penalty == 0 and self.l2Penalty != 0:
							for W in self.parameters():
								l2RegLoss += W.norm(2)**2
							loss = loss + 0.5 * self.l2Penalty * l2RegLoss
						elif self.l1Penalty != 0 and self.l2Penalty != 0:
							for W in self.parameters():
								l2RegLoss += W.norm(2)**2
								l1RegLoss += W.norm(1)
							loss = loss + self.l1Penalty * l1RegLoss + 0.5 * self.l2Penalty * l2RegLoss
					
					error.append(loss.item())

					# Backward and optimize
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

				self.epochError.append(np.mean(error))
				if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
					print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs, self.epochError[-1]))
			
			#variable to store pre-trained weight and bias
			if d <len(self.hLayer)-1:
				preW=[]
				preB=[]
				for h in range(len(hidden_layer)):
					preW.append(self.hidden[h].weight)
					preB.append(self.hidden[h].bias)
				preW.append(self.out.weight)
				preB.append(self.out.bias)

		#now set requires_grad =True for all the layers
		for l in range(len(hidden_layer)):			
			self.hidden[l].weight.requires_grad=True			
			self.hidden[l].bias.requires_grad=True
			
		self.out.weight.requires_grad=True
		self.out.bias.requires_grad=True
		
		if verbose:
			print('Pre-training is done.')

	def postTrain(self,dataLoader,learningRate,batchSize,numEpochs,verbose):

		# set device
		device = self.device
		
		# set loss function
		if self.errorFunc.upper() == 'CE':
			criterion = nn.CrossEntropyLoss()
		elif self.errorFunc.upper() == 'BCE':
			criterion = nn.BCELoss()
		elif self.errorFunc.upper() == 'MSE':
			criterion = nn.MSELoss()

		# set optimization function
		optimizer = torch.optim.Adam(self.parameters(),lr=learningRate,amsgrad=True)

		# Load the model to device
		self.to(device)
		
		# Start training
		if verbose:
			print('Training network:',self.inputDim,'-->',self.hLayerPost,'-->',self.inputDim)
		for epoch in range(numEpochs):
			error=[]
			for i, (trInput, trOutput) in enumerate(dataLoader):  
				# Move tensors to the configured device
				trInput = trInput.to(device)
				trOutput = trOutput.to(device)

				# Forward pass
				outputs = self.forwardPost(trInput)
				loss = criterion(outputs, trOutput)
				
				# Check for regularization
				if self.l1Penalty != 0 or self.l2Penalty != 0:
					l1RegLoss,l2RegLoss = torch.tensor([0.0],requires_grad=True).to(device), torch.tensor([0.0],requires_grad=True).to(device)
					if self.l1Penalty != 0 and self.l2Penalty == 0:
						for W in self.parameters():
							l1RegLoss += W.norm(1)
						loss = loss + self.l1Penalty * l1RegLoss
					elif self.l1Penalty == 0 and self.l2Penalty != 0:
						for W in self.parameters():
							l2RegLoss += W.norm(2)**2
						loss = loss + 0.5 * self.l2Penalty * l2RegLoss
					elif self.l1Penalty != 0 and self.l2Penalty != 0:
						for W in self.parameters():
							l2RegLoss += W.norm(2)**2
							l1RegLoss += W.norm(1)
						loss = loss + self.l1Penalty * l1RegLoss + 0.5 * self.l2Penalty * l2RegLoss
				
				error.append(loss.item())

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			self.epochError.append(np.mean(error))
			if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
				print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs, self.epochError[-1]))


	def fit(self,trData,trLabels,learningRate=0.001,miniBatchSize=100,numEpochsPreTrn=25,
		numEpochsPostTrn=100,standardizeFlag=True,preTraining=True,cudaDeviceId=0,verbose=False):

		# set device
		self.device = torch.device('cuda:'+str(cudaDeviceId))

		if standardizeFlag:
		#standardize data
			mu,sd,trData = standardizeData(trData)
			self.trMu=mu
			self.trSd=sd

		#create target: centroid for each class
		target=createOutputAsCentroids(trData,trLabels)

		#Prepare data for torch
		trDataTorch=Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(target).float())
		dataLoader=Data.DataLoader(dataset=trDataTorch,batch_size=miniBatchSize,shuffle=True)

		#layer-wise pre-training
		if preTraining:
			print('Running layer-wise pre-training.')
			self.preTrain(dataLoader,learningRate,miniBatchSize,numEpochsPreTrn,verbose)
		else:
			#initialize the network weight and bias
			self.initNet(self.inputDim,self.hLayerPost)
		#post training
		print('Running post-training.')
		self.postTrain(dataLoader,learningRate,miniBatchSize,numEpochsPostTrn,verbose)
		
	def predict(self,x):
		if len(self.trMu) != 0 and len(self.trSd) != 0:#standarization has been applied on training data so apply on test data
			x = standardizeData(x,self.trMu,self.trSd)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		x=torch.from_numpy(x).float().to(device)
		fOut=[x]
		with torch.no_grad():#we don't need to compute gradients (for memory efficiency)
			for l in range(len(self.hidden)):
				if self.hActFuncPost[l].upper()=='SIGMOID':
					fOut.append(torch.sigmoid(self.hidden[l](fOut[-1])))
				elif self.hActFuncPost[l].upper()=='TANH':
					fOut.append(torch.tanh(self.hidden[l](fOut[-1])))
				elif self.hActFuncPost[l].upper()=='RELU':
					fOut.append(torch.relu(self.hidden[l](fOut[-1])))
				else:#default is linear				
					fOut.append(self.hidden[l](fOut[-1]))

			if self.oActFunc.upper()=='SIGMOID':
				fOut.append(torch.sigmoid(self.out(fOut[-1])))
			else:
				fOut.append(self.out(fOut[-1]))

		return fOut
