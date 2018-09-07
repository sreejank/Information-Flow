
#Compute the Transfer Entropy Matrix from a Subject's fMRI Data (parcellated with the Shen Atlas)
from jpype import *
from load import *
import pandas as pd
from pandas import DataFrame
from joblib import Parallel, delayed
import pickle
import os
import math 
import time

#width=".25" #..0001, .001, .01, .1, .25, .5
#kraskov_k="6"
#Calculate transfer entropy fromData toData (numpy arrays) using KSG Estimation methodology. Gives information in: nats
def getTransferEntropy(teCalcClass,teCalc,fromData,toData):
	
	teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD,
			teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
	
	#teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, kraskov_k)
	#teCalc.initialise()
	#teCalc.setObservations(fromData, toData)
	
	optimisedK = int(teCalc.getProperty(teCalcClass.K_PROP_NAME))
	optimisedKTau = int(teCalc.getProperty(teCalcClass.K_TAU_PROP_NAME))
	optimisedL = int(teCalc.getProperty(teCalcClass.L_PROP_NAME))
	optimisedLTau = int(teCalc.getProperty(teCalcClass.L_TAU_PROP_NAME))
	
	#teCalc.setProperty("KERNEL_WIDTH",width)
	teCalc.initialise()
	teCalc.setObservations(fromData,toData)
	transfer_entropy=teCalc.computeAverageLocalOfObservations()
	return transfer_entropy

#Calculate Shannon Entropy using Kernel Estimation on data. Gives information in: bits
def getShannonEntropy(seCalc,data):
	seCalc.setProperty("KERNEL_WIDTH",width)
	seCalc.initialise()
	seCalc.setObservations(data)
	return seCalc.computeAverageLocalOfObservations()

def buildTransferEntropyMatrix(fname,debug=False):
	#print("Starting JVM and importing Java Objects...")
	print(fname)
	
	teCalcClass = JPackage("infodynamics.measures.continuous.gaussian").TransferEntropyCalculatorGaussian
	teCalc = teCalcClass()
	seCalcClass= JPackage("infodynamics.measures.continuous.kernel").EntropyCalculatorKernel
	seCalc= seCalcClass()
	#print("Success!")

	raw_df=pd.read_table(fname,delimiter='\t')
	data=DataFrame.as_matrix(raw_df)[:,1:-1]
	mat=np.negative(np.ones((data.shape[1],data.shape[1])))
	print("Building matrix for "+fname)
	for i in range(mat.shape[0]):
		if debug:
			print("Row: "+str(i)+", Time: "+str(time.time()))
		for j in range(mat.shape[1]):
			if i!=j:
				c1=data[:,i]
				c2=data[:,j]
				mat[i,j]=getTransferEntropy(teCalcClass,teCalc,c1,c2)
			else:
				c=data[:,i]
				mat[i,j]=getShannonEntropy(seCalc,c) #Keep in bits. 
	if data.shape[1]==265:
		mat=np.insert(mat,(25,132,176),0.0,axis=0)
		mat=np.insert(mat,(25,132,176),0.0,axis=1)
	#shutdownJVM()
	return mat

def calculateTaskMatrices():
	fnames=[]
	for fname in os.listdir('data/roimean_fullruns'):
		if fname.endswith('_FullTask_matrix_bis_matrix_roimean.txt'):
			fnames.append(str(os.path.join('data/roimean_fullruns',fname)))
	task_matrices=Parallel(n_jobs=20)(delayed(buildTransferEntropyMatrix)(fname) for fname in fnames)
	print("Dumping data...")

	pickle.dump(task_matrices,open(r'data/gradCPT_task_mats_entropy.pickle',"wb"))

def calculateRestMatrices():
	fnames=[]
	for fname in os.listdir('data/roimean_fullruns'):
		if fname.endswith('_FullRest_matrix_bis_matrix_roimean.txt'):
			fnames.append(str(os.path.join('data/roimean_fullruns',fname)))
	rest_matrices=Parallel(n_jobs=1)(delayed(buildTransferEntropyMatrix)(fname) for fname in fnames)
	print("Dumping data...")

	pickle.dump(rest_matrices,open(r'data/gradCPT_rest_mats_entropy_gaussian.pickle',"wb"))



#jarLocation = "/nexsan/home01/chun/sk2436/infodynamics-dist-1.3.1/infodynamics.jar"
print("Starting JVM")
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation,"-l h_vmem=100m")
#buildTransferEntropyMatrix('data/roimean_fullruns/7329_FullRest_matrix_bis_matrix_roimean.txt',debug=True)
calculateTaskMatrices()
calculateRestMatrices()
