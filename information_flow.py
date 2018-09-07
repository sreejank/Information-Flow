# Compute Information Flow Functional Connectomes using Transfer Entropy matrices. 
from load import *
import networkx as nx
import pickle
from joblib import Parallel, delayed
from scipy.io import loadmat
import sys
import pickle

#Do: 0.01, .25, 5, kraskov
threshold='special'
grouping='none'
kernel='kraskov'

network_map={}
lobe_map={}

#Build Node to Network Map
raw_node_list=np.genfromtxt('data/shen_atlas.csv',delimiter=',')[1:,1]
for i in range(268):
	network_map[i]=raw_node_list[i]-1

#Build Node to Lobe Map
lobes=['R_prefrontal','R_motor','R_insula','R_parietal','R_temporal','R_occipital','R_limbic','R_cerebellum','R_subcortex','R_brainstem',
'L_prefrontal','L_motor','L_insula','L_parietal','L_temporal','L_occipital','L_limbic','L_cerebellum','L_subcortex','L_brainstem']




lobe_map[0] = list(range(22))
lobe_map[1] = list(range(22,33))
lobe_map[2] = list(range(33,37))
lobe_map[3] = list(range(37,50))
lobe_map[4] = list(range(50,72))
lobe_map[5]= list(range(72,82))
lobe_map[6] = list(range(82,99))
lobe_map[7]= list(range(99,119))
lobe_map[8]=list(range(119,128))
lobe_map[9]=list(range(128,133))
lobe_map[10]=list(range(133,157))
lobe_map[11]=list(range(157,167))
lobe_map[12]=list(range(167,170))
lobe_map[13]=list(range(170,184))
lobe_map[14]=list(range(184,202))
lobe_map[15]=list(range(202,216))
lobe_map[16] =list(range(216,235))
lobe_map[17]= list(range(235,256))
lobe_map[18] = list(range(256,264))
lobe_map[19] = list(range(264,268))

vars_adhd=loadmat('/nexsan/home01/chun/sk2436/Downloads/adhd_node_map.mat')
nodemap=[x[0] for x in vars_adhd['node_map']-1]




def lobemask(n1,n2):
	d=lobe_map
	group1=lobe_map[n1]
	group2=lobe_map[n2]
	mask=np.zeros((268,268))
	for i in group1:
		for j in group2:
			mask[i,j]=1.0
	for i in group2:
		for j in group1:
			mask[i,j]=1.0
	return mask



lobe_group_masks=np.zeros((20,20),dtype=object)
for i in range(20):
	for j in range(20):
		lobe_group_masks[i,j]=lobemask(i,j)


def sparsify(mat):
	mat2=mat.copy()

	vector=[]
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			if i!=j:
				element=(mat[i,j],i,j)
				vector.append(element)
			else:
				mat2[i,j]=0.0
	sorted_vector=sorted(vector,key=lambda x: x[0])
	n=len(sorted_vector)
	num_deleted=0
	if threshold!='special':
		while num_deleted<int(n*(1.0-threshold)):
			element=sorted_vector[num_deleted]
			mat2[element[1],element[2]]=0.0
			num_deleted+=1
	for i in range(mat2.shape[0]):
		for j in range(mat2.shape[1]):
			if mat2[i,j]<0:
				mat2[i,j]=0.0
	return mat2 


def capacitate(G,shannonEntropies):
	H=G.copy()
	weights={}
	for edge in G.edges():
		weights[edge]=G[edge[0]][edge[1]]['weight']

	for v in G.nodes():
		v_in=str(v)+'_in'
		H.add_node(v_in)
		v_out=str(v)+'_out'
		H.add_node(v_out)

	for v in G.nodes():
		v_in=str(v)+'_in'
		v_out=str(v)+'_out'
		preds=G.predecessors(v)
		succ=G.successors(v)
		H.remove_node(v)
		for pred in preds:
			tup=(pred,v)
			#print("Adding edge "+str(pred)+'_out'+" to "+v_in+" with weight "+str(weights[tup]))
			H.add_edge(str(pred)+'_out',v_in,weight=weights[tup])
		for successor in succ:
			tup=(v,successor)
			#print("Adding edge "+v_out+" to "+str(successor)+'_in'+" with weight "+str(weights[tup]))
			H.add_edge(v_out,str(successor)+'_in',weight=weights[tup])
		H.add_edge(v_in,v_out,weight=shannonEntropies[v])
	return H
mats=[]
from networkx.convert_matrix import from_numpy_matrix


def getFlowGrouping(mat):
	n=20
	shannonEntropies=[mat[i,i] for i in range(mat.shape[0])]
	mat2=sparsify(mat)
	connectome=np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			#print("Nodes "+str(i)+" "+str(j)+".")
			G=nx.DiGraph(data=np.multiply(mat2,lobe_group_masks[i,j]))
			
			remove=[node for node,degree in G.degree().items() if degree==0]
			G.remove_nodes_from(remove)
			flow=0.0
			original_nodes=G.nodes()
			#H=capacitate(G,shannonEntropies)
			for node1 in original_nodes:
				for node2 in original_nodes:
					if node1!=node2:
						if node1 in lobe_map[i] and node2 in lobe_map[j]:
							flow+=nx.maximum_flow_value(G,node1,node2,capacity='weight')
							#flow+=nx.maximum_flow_value(H,str(node1)+'_in',str(node2)+'_out',capacity='weight')
			connectome[i,j]=flow
			
	return connectome





def calculateEntropygradCPT(cores):
	print("Calculating flows for gradCPT "+fend)
	with open('Entropy/Connectivity/gradCPT_task_mats_entropy.pickle','rb') as p:
		gradCPT_task_mats_entropy=np.asarray(pickle.load(p))
	with open('Entropy/Connectivity/gradCPT_rest_mats_entropy.pickle','rb') as p:
		gradCPT_rest_mats_entropy=np.asarray(pickle.load(p))

	if grouping=='lobe':
		gradCPT_task_flows_entropy=[getFlowGrouping3(gradCPT_task_mats_entropy[i,:,:]) for i in range(25)]
		gradCPT_rest_flows_entropy=[getFlowGrouping3(gradCPT_rest_mats_entropy[i,:,:]) for i in range(25)]
	elif grouping=='network':
		gradCPT_task_flows_entropy=Parallel(n_jobs=cores)(delayed(getFlowNetworks)(gradCPT_task_mats_entropy[i,:,:]) for i in range(25))
		gradCPT_rest_flows_entropy=Parallel(n_jobs=cores)(delayed(getFlowNetworks)(gradCPT_rest_mats_entropy[i,:,:]) for i in range(25))
	else:
		gradCPT_task_flows_entropy=Parallel(n_jobs=cores)(delayed(getFlowNoGrouping)(gradCPT_task_mats_entropy[i,:,:]) for i in range(25))
		gradCPT_rest_flows_entropy=Parallel(n_jobs=cores)(delayed(getFlowNoGrouping)(gradCPT_rest_mats_entropy[i,:,:]) for i in range(25))
		#gradCPT_task_flows_entropy=[getFlowNoGrouping(gradCPT_task_mats_entropy[i,:,:]) for i in range(25)]
		#gradCPT_rest_flows_entropy=[getFlowNoGrouping(gradCPT_rest_mats_entropy[i,:,:]) for i in range(25)]
	
	print("Dumping data...")
	pickle.dump(gradCPT_task_flows_entropy,open('Entropy/Flow/gradCPT_task_flows_entropy3'+fend,'wb'))
	pickle.dump(gradCPT_rest_flows_entropy,open('Entropy/Flow/gradCPT_rest_flows_entropy3'+fend,'wb'))
	print("Done!")


calculateEntropygradCPT(10)

