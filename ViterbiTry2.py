#import pdb
#pdb.set_trace()

#An implementation of viterbi decoding (http://en.wikipedia.org/wiki/Viterbi_decoder)
#This is acceptable because our grammar only has around 10 POS. Viterbi decoding will be slow for higher POS numbers
#Quick explanation of Viterbi decoding:
#We have an MRF like this:
#  word1         word2         word3      ....
#  x_1_1  \      x_2_1  \      x_3_1  \      
#  x_1_2   |->   x_2_2   |->   x_3_2   |->....   
#  x_1_2  /      x_2_3  /      x_3_3  / 
#We traverse down words, and for each possible state of the word, take: P(curPOS)= max over prevPOS of:P(curPOS|prevPOS)*P(prevPOS)
#and we note the prevPOS at each iteration
#e.g. P(x_2_2)=max x_1_i of P(x_2_2|P(x_1_i)*P(x_1_i)
#for x_1_i=x_1_1,x_1_2,x_1_3
#Then at the end we take the max prob for the current words POS and reconstruct the entire tagging for the sentence
#Based on the prevPOS we were taking note of.

def get_tuple_list(factorLine,stateSizeArray):
#Return all possible combos of states for two words e.g. for above example it would be:
#((word1,1),word(2,1)),((word1,1),word(2,2)),((word1,1),word(2,3)),((word1,2),word(2,1))...etc
  variables=factorLine.rstrip().split()
	if not variables[0]=="2":
		print factorLine
		print "^^^error with factorLine in get_tuple_list"
	firstVar=variables[1]
	firstStateSize=int(stateSizeArray[int(firstVar)])
	secondVar=variables[2]
	secondStateSize=int(stateSizeArray[int(secondVar)])
	return [ ((firstVar,i),(secondVar,j))for i in range(0,firstStateSize) for j in range(0,secondStateSize)]

def get_cached_edges(factorLines=None):
#Get all edges (POSi,POSj) for i,j in {0,...,NumPOS}
	if factorLines==None:
		return cached_edges
	else:
		for line in factorLines:
			lineArray=line.rstrip().split()
			newEdge=(lineArray[1],lineArray[2])
			cached_edges.append(newEdge)


def return_prob_from_lines(filename,probabilityLines,factorLines,stateSizeLine):
#forms a dictionary probabilities from our MRF
#probabilities[(word1,POS1)]=log P(POS1|word1)
#probabilities[(POS1,POS2)]=log P(POS2|POS1)
	stateSizeArray=stateSizeLine.rstrip().split()
	Probabilities={}
	probLineCounter=0
	for factorLine in factorLines:
		variables=factorLine.rstrip().split()
		#print variables
		#print "Variables!"
		if variables[0]=="1":
			while (not probabilityLines[probLineCounter].isspace()): probLineCounter=probLineCounter+1
			probabilitiesArray=probabilityLines[probLineCounter+2].rstrip().split()
			#check that probability line has the same amount of entries as n x m for var1 having n states and var2 having m states
			#don't need a tuples list.. possible source of errors due to no check..
			
			if filename=="kitchen.uai" or filename=="office.uai":
				for i in range(0,len(probabilitiesArray)):
					tuple=(variables[1],i)
					#Probabilities[tuple]=math.exp(float(probabilitiesArray[i]))
					if float(probabilitiesArray[i])!=0:
						Probabilities[tuple]=math.log(float(probabilitiesArray[i]))
					else:
						Probabilities[tuple]=float("-inf")
			else:
				for i in range(0,len(probabilitiesArray)):
					tuple=(variables[1],i)
					Probabilities[tuple]=float(probabilitiesArray[i])
			probLineCounter=probLineCounter+1
		elif variables[0]=="2":
			while (not probabilityLines[probLineCounter].isspace()):
				probLineCounter=probLineCounter+1
			probabilitiesArray=[]
			if filename=="kitchen.uai" or filename=="office.uai":
				BothProbabilityLines=probabilityLines[probLineCounter+2]+probabilityLines[probLineCounter+3]
				probabilitiesArray=BothProbabilityLines.rstrip().split()
			else:
				probabilitiesArray=probabilityLines[probLineCounter+2].rstrip().split()
			tuplesList=get_tuple_list(factorLine,stateSizeArray)
			if len(tuplesList)!=len(probabilitiesArray):
				print tuplesList
				print probabilitiesArray
				print len(tuplesList)
				print len(probabilitiesArray)
				print "^^^^length of tuplesList and probabiltiesArray do not match"
			else:
			#	print "about to assign to probabilities..."
				if len(probabilitiesArray)==0:
					print "probabilitiesArray is empty..."
				if filename=="kitchen.uai" or filename=="office.uai":
					for i in range(0,len(probabilitiesArray)):
						tuple=tuplesList[i]
						#Probabilities[tuple]=math.exp(float(probabilitiesArray[i]))
						if float(probabilitiesArray[i])!=0:
							Probabilities[tuple]=math.log(float(probabilitiesArray[i]))
						else:
							Probabilities[tuple]=float("-inf")
				else:
					for i in range(0,len(probabilitiesArray)):
						tuple=tuplesList[i]
						Probabilities[tuple]=float(probabilitiesArray[i])
				probLineCounter=probLineCounter+1
		else:
			print "factorLine error!"
	return Probabilities

def Eliminate_Variable(EliminateMe,stateSizeArray,Priors,Traceback):
#Eliminates EliminateMe by constucting P(POSi)=max over j P(POSi)*P(POSi|POSj)*P(POSj)
#remember all probs are log probs, so we add, not multiply
#Also make appropriate note in traceback.
	for i in range(0,stateSizeArray[EliminateMe+1]):
		priorChoices=None
		priorChoices=[0]*stateSizeArray[EliminateMe]
		otherPriorKey=(str(EliminateMe+1),i)
		for j in range(0,stateSizeArray[EliminateMe]):
			priorKey=(str(EliminateMe),j)
			jointKey=(priorKey,otherPriorKey)
			priorChoices[j]=Priors[EliminateMe][j]+Probabilities[jointKey]+Probabilities[otherPriorKey]
		BestChoice=priorChoices.index(max(priorChoices))
		#print BestChoice
		#print EliminateMe
		#print Traceback[EliminateMe+1][i]
		Traceback[EliminateMe+1][i]=Traceback[EliminateMe][BestChoice]+((EliminateMe,BestChoice),)
		Priors[EliminateMe+1][i]=priorChoices[BestChoice]
		

def Viterbi(lines,filename):
#Form auxiliary variables, then iterate over sentence in order, eliminating vars as you go
#And keep the traceback
#At the end, take max over probabilities for that words state, and use traceback of that state as the
#Tagging of your sentence
	#f=open(filename,'r')
	#lines=f.readlines()
	global cached_edges
	cached_edges=[]
	global Probabilities
	endFactorLines=int(lines[0].rstrip())
	endPriorLines=int(lines[1].rstrip())
	probabilityLines=lines[endFactorLines:len(lines)]
	#for line in probabilityLines: print line
	factorLines=lines[6:endFactorLines]
	edgeLines=lines[endPriorLines:endFactorLines]
	#for line in lines: print line
	get_cached_edges(edgeLines)
	stateLine=lines[4]
	Probabilities=return_prob_from_lines(filename,probabilityLines,factorLines,stateLine)
	#for key in Probabilities.keys(): print key
	stateSizeArray=lines[4].rstrip().split()
	stateSizeArray=[int(stateSizeEle) for stateSizeEle in stateSizeArray]
	numVariables=len(stateSizeArray)-1
	EliminationOrder=[0]*(numVariables)
	global Priors
	global Traceback
	maxStates=max(stateSizeArray)
	Priors=[[0]*maxStates for i in range(0,numVariables+1)]
	for i in range(0,stateSizeArray[0]):
		Priors[0][i]=Probabilities[('0',i)]
	Traceback=[[()]*(maxStates) for i in range(0,numVariables+1)]
	#for i in range(0,stateSizeArray[0]):
	#	Traceback[0][i]=((0,i),)
	for i in range(0,numVariables):#then elimination order is 1,2,3, etc.
		EliminationOrder[i]=i
	for i in range(0,len(EliminationOrder)):
		#print i
		Eliminate_Variable(EliminationOrder[i],stateSizeArray,Priors,Traceback)
	#for i in range(0,len(EliminationOrder)-1):
	#	print Priors[i][:]
	#	print Traceback[i][:]
	BestValue=float("-inf")
	BestChoice=-1
	for i in range(0,maxStates-1):
		#print Priors[numVariables-1]
		#print i
		#print "here"
		if Priors[numVariables-1][i]>BestValue or BestValue==float("-inf"):
			BestChoice=i
			BestValue=Priors[numVariables-1][i]
			
	#print "Best choice: "+str(BestChoice)
	#print Priors
	MAPTuple=Traceback[numVariables-1][BestChoice]
	MAPAssignment={}
	#print MAPTuple
	for tup in MAPTuple:
		MAPAssignment[int(tup[0])]=tup[1]
	MAPAssignment[numVariables-1]=BestChoice
	if BestChoice==-1:
		print "gah..."
	#print MAPTuple
	#print Traceback
	#print MAPAssignment
	#print "MAPASSIGN"
	return MAPAssignment
	
#Viterbi("MRF1-10")
