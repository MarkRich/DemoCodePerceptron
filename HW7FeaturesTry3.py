import numpy
import math
import pdb
from multiprocessing import Process
#import PGMHW5try4
import matplotlib.pyplot as plt
import ViterbiTry2
#pdb.set_trace()
numpy.set_printoptions(threshold=numpy.nan )


#An implementation of a feature-rich perceptron from this homework set: http://cs.nyu.edu/~dsontag/courses/pgm13/assignments/ps7.pdf
#
#Requires numpy and pyplot. If you do not have pyplot just search for "plt" and comment out the lines that use it
#
#
#
def readSentence(filename):
  sentences=open(filename).readlines()
	SentenceDict={}
	for i in range(0,len(sentences)):#For each word in sentence, split it up into array by comma.
		SentenceDict[i]=sentences[i].rstrip().split(',')
	#Each sentence is a group of n words on n lines, each word is formatted as a csv line
	return SentenceDict


def get_Feature_Vector(word,i,sentence,FeatVectorSize,FeatStartPoints,knownWords,wordPairs):
#Given word, return shortened feature vector representing that word.
#To form the real feature vector, we offset the shorter feature vector based on its POS tag
#The following 30 lines just sets the binary feature vector to 0 or 1 based on observed data, for each index.
	FeatVect=[0]*FeatVectorSize
	FeatVect[0]=1#For bias
	for j in range(2,6):
		FeatVect[FeatStartPoints[j-2]+int(sentence[i][j+1])]=1
	if special_chars(word):
		#print word
		#print "Special chars"
		FeatVect[FeatStartPoints[4]+1]=1
	else:
		FeatVect[FeatStartPoints[4]]=1
	if hyphen(word):
		#print word
		#print "hyphen"
		FeatVect[FeatStartPoints[5]+1]=1
	else:
		FeatVect[FeatStartPoints[5]]=1
	if digits(word):
		#print word
		#print "digits"
		FeatVect[FeatStartPoints[6]+1]=1
	else:
		FeatVect[FeatStartPoints[6]]=1
	if word in knownWords:
		FeatVect[FeatStartPoints[7]+1]=1
	else:
		FeatVect[FeatStartPoints[7]]=1
	if i!=0 and i!=len(sentence)-1:
		wordPair=sentence[i-1][0]+"|"+sentence[i+1][0]
		if wordPair in wordPairs:
			FeatVect[FeatStartPoints[8]+wordPairs[wordPair]]=1
		else:
			FeatVect[FeatStartPoints[8]]=1
	return FeatVect
	
	

def get_Feature_Vector_Total(Result,sentence,FeatVectorSize,FeatStartPoints,TransitionProbStart,knownWords,wordPairs):
	#Takes a tagging of a sentence (Result), the sentence itself, and a few auxiliary variables to form feature vector for each word
	#And forms the sentences feature vector. The sentences feature vector is the sum of all the feature vectors of the words
	#The words feature vector are the shortened feature vectors from get_Feature_Vector, offset appropriately with their POS tagging
	#from result
	TotalVect=numpy.zeros(FeatVectorSize*10+10*10)
	for i in range(0,len(sentence)-1):
		tempVect=get_Feature_Vector(sentence[i],i,sentence,FeatVectorSize,FeatStartPoints,knownWords,wordPairs)
		tempFeatVect=numpy.zeros(FeatVectorSize*10+10*10)
		yHat=Result[i]
		sLabelPoint=yHat*FeatVectorSize
		eLabelPoint=(yHat+1)*FeatVectorSize
		tempFeatVect[sLabelPoint:eLabelPoint]=tempVect
		if i!=0 and i!=len(sentence)-1:
			tempFeatVect[TransitionProbStart+Result[i-1]*10+yHat]=1
		TotalVect=TotalVect+tempFeatVect
	return TotalVect

def get_Feature_Vector_Total_Y(sentence,FeatVectorSize,FeatStartPoints,TransitionProbStart,knownWords,wordPairs):
#Does the same as get_Feature_Vector total, but instead of using result as our tagging, we use the actual tagging of the sentence.
#This return the true feature vector for a sentence
	TotalVect=numpy.zeros(FeatVectorSize*10+10*10)
	for i in range(0,len(sentence)-1):
		tempVect=get_Feature_Vector(sentence[i],i,sentence,FeatVectorSize,FeatStartPoints,knownWords,wordPairs)
		tempFeatVect=numpy.zeros(FeatVectorSize*10+10*10)
		y=int(sentence[i][1])-1
		sLabelPoint=y*FeatVectorSize
		eLabelPoint=(y+1)*FeatVectorSize
		tempFeatVect[sLabelPoint:eLabelPoint]=tempVect
		if i!=0 and i!=len(sentence)-1:
			tempFeatVect[TransitionProbStart+(int(sentence[i-1][1])-1)*10+y]=1
		TotalVect=TotalVect+tempFeatVect
	return TotalVect



def preprocess_Features(NumTraining,minTraining):
	#Need to make:
	#FeatVectorSize, TransitionProbStart, FeatStartPoints, NumFeatures, UsePrefix,UseSuffix
	#	Bias,firstcap,capitalized,specialchar,digits,hyphens,prefix2,prefix3,suffix2,suffix3
	#add unknown?

	seenPairs={}
	seenPairsDist={}
	wordCounts={}
	#	Bias,firstcap,capitalized,specialchar,digits,hyphens,prefix2,prefix3,suffix2,suffix3
	
	#The below makes a dictionary mapping words pairs-> appearance counts
	#The word pairs are the adjacent words for each word.
	#E.g. if our sentence is word1 word2 word3 word4, we get word pairs: word1|word3 and word2|word4
	#Also accumulates a count called seenPairsDist which, for each wordPair, gives a distribution of the (not used any more)
	for i in range(minTraining,NumTraining*100):
		sentence=readSentence('Data/train-'+str(i)+'.txt')
		for w in range(0,len(sentence)):
			if w!=0 and w!=len(sentence)-1:
				#print len(sentence)
				#print w
				OtherWords=sentence[w-1][0]+"|"+sentence[w+1][0]
				if not OtherWords in seenPairs.keys():
					seenPairs[OtherWords]=1
					seenPairsDist[OtherWords]=[0]*10
					seenPairsDist[OtherWords][int(sentence[w][1])-1]=1
				else:
					seenPairs[OtherWords]+=1
					seenPairsDist[OtherWords][int(sentence[w][1])-1]+=1
				if not sentence[w][0] in wordCounts.keys():
					wordCounts[sentence[w][0]]=1
				else:	
					wordCounts[sentence[w][0]]+=1
	knownWords=[]
	usePairs={}
	PairsCounter=1
	#Fill knownWords with all words seen more than once
	#Fill usePairs with all word pairs seen more than five times
	for word in wordCounts.keys():
		#print word+":"+str(wordCounts[word])
		if wordCounts[word]>1:
			knownWords.append(word)
	for wordPair in seenPairs.keys():
		if (seenPairs[wordPair]>5):
			#print wordPair+":"+str(seenPairs[wordPair])+"--->"+str(seenPairsDist[wordPair])
			usePairs[wordPair]=PairsCounter
			PairsCounter+=1
				

	
	FeatVectorSize=1+2+2+201+201+2+2+2+2+len(usePairs)
	#1 for bias, each 2 is a binary features, the 201s are for prefix/suffix, and then a number for each wordPair we use
	FeatStartPoints=[1,3,4,4+201,4+201+201,6+201+201,8+201+201,10+201+201,12+201+201]
	#Our offset to access each feature we want to set on/off
	TransitionProbStart=FeatVectorSize*10 #=FeatVectorSize*NumPOS
	#TransitionProbStart is the index where transition probabilities start as opposed to POS tags.
	return (FeatVectorSize,TransitionProbStart,FeatStartPoints,usePairs,knownWords)


def special_chars(word):
	if type(word) is list:
		word=word[0]
	if "*" in word or "&" in word or "%" in word or "#" in word or "@" in word:
		return True
	else:
		return False
	
def hyphen(word):
	if type(word) is list:
		word=word[0]
	if "-" in word:
		return True
	else:
		return False

def digits(word):
	if type(word) is list:
		word=word[0]
	if "0" in word or "1" in word or "2" in word or "3" in word or "4" in word or "5" in word \
		or "6" in word or "7" in word or "8" in word or "9" in word:
		return True
	else:
		return False

	
def run_and_test(minTraining,NumTraining):	
	NumPOS=10
	print "preprocessesing "+str(NumTraining)+"00 sentences"
	BaseInformationTup=preprocess_Features(NumTraining,minTraining)#Get initialization info
	print "done preprocessesing "+str(NumTraining)+"00 sentences"
	FeatVectorSize=BaseInformationTup[0]
	TransitionProbStart=BaseInformationTup[1] 
	#^The index that indicates where transition probabilities start (e.g. feat vect elements giving p(noun1->noun2), p(noun1->verb2) transition agreements (they're not probabilities), as opposed to p(noun1|word1)
	FeatStartPoints=BaseInformationTup[2]
	#For p(Part of Speech|word), gives the offsets for where we can reach the appropriate index for each feature (e.g. has hyphen)
	wordPairs=BaseInformationTup[3]
	#All adjacent wordPairs which we consider as a feature
	knownWords=BaseInformationTup[4]
	#All words that we have seen (for knownWord feature)
	W=numpy.zeros(TransitionProbStart+10*10)
	#Our feature vector
	Wbar=numpy.zeros(TransitionProbStart+10*10)
	#Our averaged feature vector
	LabelError=0
	TransitionError=0
	totalRepetitions=NumTraining-minTraining+1
	AmountCorrect=[0]*totalRepetitions
	WordsWrong={}
	for numRepititions in range(0,50):#50 rounds of training on corpus
		numTotal=0
		numCorrect=0
		for t in range(minTraining,100*NumTraining):#Iterate over training sentences
			if numCorrect==0 and t==100*NumTraining-1: #none correct at last repetition
				#print str(NumTraining)+"--->"+str(numRepititions)+":"+str(t)+'0'
				print "for "+str(t)+" training sentences, at round "+str(numRepititions)+" 0 correct"
			elif numCorrect!=0 and t==100*NumTraining-1:#else print % correct
				print "for "+str(t)+" training sentences, at round "+str(numRepititions)+" "+str(float(numCorrect)/float(numTotal))+" correct"
				#print str(NumTraining)+"--->"+str(numRepititions)+":"+str(t)+"-->"+str(float(numCorrect)/float(numTotal))
			sentence=readSentence('Data/train-'+str(t)+'.txt')
			PrevWrong=True
			Result={}
			tempString=str(int(sentence[0][1])-2)+" ," #Not used..
			for i in range(1,len(sentence)):tempString=tempString+str(int(sentence[i][1])-1)+" ,"#Not used...
			#if PrevWrong or i==0:
			if not Result:#Given current weight vector, sentence, etc., get POS labeling of sentence using W to describe joints/priors
			#and use viterbi decoding on resulting MRF
				Result=write_MRF_get_Label(W,sentence,minTraining,NumTraining,NumPOS,TransitionProbStart,FeatVectorSize,FeatStartPoints,knownWords,wordPairs)
				PrevWrong=True
			for i in range(0,len(sentence)-2):#Count number of correct POS taggings
				numTotal=numTotal+1
				curWord=sentence[i]
				yHat=Result[i]
				yHatNext=Result[i+1]
				yNext=int(sentence[i+1][1])-1
				y=int(curWord[1])-1
				if y==yHat:
					numCorrect=numCorrect+1
					if yNext==yHatNext:
						PrevWrong=False
				else:
					if not curWord[0] in WordsWrong:
						WordsWrong[curWord[0]]=1
					else:
						WordsWrong[curWord[0]]+=1
			#Get feature vector of correct tagging and our guessed tagging, take their averaged difference as update
			tempXGuessLabel=get_Feature_Vector_Total(Result,sentence,FeatVectorSize,FeatStartPoints,TransitionProbStart,knownWords,wordPairs)
			tempXCorrectLabel=get_Feature_Vector_Total_Y(sentence,FeatVectorSize,FeatStartPoints,TransitionProbStart,knownWords,wordPairs)
			W=W+tempXCorrectLabel-tempXGuessLabel
			Wbar=Wbar+ W/(50*100*NumTraining)
		#print WordsWrong
	NumCorrectTest=0
	NumTotalTest=0
	for t in range(1,1000):# For each test sentence, read it, tag it, and count number correct
		sentence=readSentence('Data/test-'+str(t)+'.txt')
		NumTotalTest=NumTotalTest+len(sentence)-1
		for i in range(0,len(sentence)-1):
			curWord=sentence[i]
			if i==0:
				Result=write_MRF_get_Label(Wbar,sentence,minTraining,NumTraining,NumPOS,TransitionProbStart,FeatVectorSize,FeatStartPoints,knownWords,wordPairs)
			yHat=Result[i]
			y=int(curWord[1])-1
			if y==yHat:
				NumCorrectTest=NumCorrectTest+1
	NumCorrectTrain=0
	NumTotalTrain=0
	# For each train sentence, read it, tag it, and count number correct
	for t in range(1,100*NumTraining):
		sentence=readSentence('Data/train-'+str(t)+'.txt')
		NumTotalTrain=NumTotalTrain+len(sentence)-1
		for i in range(0,len(sentence)-1):
			curWord=sentence[i]
			if i==0:
				Result=write_MRF_get_Label(Wbar,sentence,minTraining,NumTraining,NumPOS,TransitionProbStart,FeatVectorSize,FeatStartPoints,knownWords,wordPairs)
			yHat=Result[i]
			y=int(curWord[1])-1
			if y==yHat:
				NumCorrectTrain=NumCorrectTrain+1
	print "========"#Print results of tagging train/test sentences.
	print "Finished "+str(100*NumTraining)+" sentences"
	print "--->Amount correct test: "+str(float(NumTotalTest-NumCorrectTest)/float(NumTotalTest))	
	print "--->Amount correct train: "+str(float(numCorrect)/float(numTotal))
	for i in range(0,len(W),10):
		print W[i:i+9]
	print "========"
	AmountCorrect[NumTraining-1]=float(NumCorrectTest)/float(NumTotalTest)
	f=open("numCorrectRegularFeat.txt","a+")
	b=open("numCorrectTrainFeat.txt","a+")
	f.write("NumTraining:"+str(NumTraining)+str(float(NumTotalTest-NumCorrectTest)/float(NumTotalTest))+" \n")
	b.write("NumTraining:"+str(NumTraining)+str(float(NumTotalTrain-NumCorrectTrain)/float(NumTotalTrain))+" \n")

def write_MRF_get_Label(W,sentence,minTraining,NumTraining,NumPOS,TransitionProbStart,FeatVectorSize,FeatStartPoints,knownWords,wordPairs):
	#Based on W, our weight vector, we write an MRF in UAI format (http://www.cs.huji.ac.il/project/PASCAL/fileFormat.php)
	lineCounter=0
	#Next 10 lines write the preamble
	lines=[]
	lines.append(str(6+2*len(sentence)-1)+"\n")
	lines.append(str(6+len(sentence))+"\n")
	lines.append("Markov\n")
	lines.append(str(len(sentence))+"\n")
	lines.append("10 "*(len(sentence))+"\n")
	lines.append(str(len(sentence)+len(sentence)-1)+"\n")
	for j in range(0,len(sentence)):
		lines.append("1 "+str(j)+"\n")
	for j in range(0,len(sentence)-1):
		lines.append("2 "+str(j)+" "+str(j+1)+"\n")
	for j in range(0,len(sentence)):
		#For each word in the sentence, perform dot(FeatVect,WeightVect) for appropriate offset based on POS
		#To form the priors for each words (e.g. P(POS|word))
		ProbVector=[0]*NumPOS
		x=get_Feature_Vector(sentence[j],j,sentence,FeatVectorSize,FeatStartPoints,knownWords,wordPairs)
		for k in range(0,NumPOS):
			startPoint=k*FeatVectorSize
			endPoint=(k+1)*FeatVectorSize
			ProbVector[k]=sum(W[startPoint:endPoint]*x)#dot product
		ProbSum=sum(ProbVector)
		ProbString=str(ProbVector[0])
		for k in range(1,NumPOS):
			ProbString=ProbString+" "+str(ProbVector[k])
		ProbString=ProbString+"\n"
		lines.append(" \n")
		lines.append(str(NumPOS)+"\n")
		lines.append(ProbString)
	ProbVector=[0]*(NumPOS*NumPOS)
	ProbVector=W[TransitionProbStart:len(W)]
	ProbString=str(ProbVector[0])
	for j in range(1,NumPOS*NumPOS):
		#For each POS1->POS2, we write P(POS2|POS1)
		ProbString=ProbString+" "+str(ProbVector[j])
	ProbString=ProbString+"\n"
	for j in range(0,len(sentence)-1):
		lines.append(" \n")
		lines.append(str(NumPOS*NumPOS)+"\n")
		lines.append(ProbString)
	#Once we have written the MRF, use viterbi decoding to get POS tagging
	Result1=ViterbiTry2.Viterbi(lines,"MRF"+str(minTraining)+"-"+str(NumTraining))
	Dual_Wrong=0
	sum_Wrong=0
	return Result1

		
			
def run_Perceptron(): #The main handle of the program
	#trains numProcesses perceptrons on (offsetTrainingSentences+NumTraining+1)*100 training sentences
	#then evaluates the perceptrons performance on 1000 test sentences
	#Use more processes if you have a cluster.
	numProcesses=2
	offSetTrainingSentences=15
	Processes=[None]*numProcesses
	global AmountCorrect
	AmountCorrect=[0]*numProcesses
	for NumTraining in range(0,numProcesses):
		Processes[NumTraining]=Process(target=run_and_test,args=(1,offSetTrainingSentences+NumTraining+1,))
	for NumTraining in range(0,numProcesses):
		Processes[NumTraining].start()
	for NumTraining in range(0,numProcesses):
		Processes[NumTraining].join()
	plt.plot(AmountCorrect)
	plt.show()
	#run_and_test(1,1)
	

if __name__=="__main__":
	run_Perceptron()
