import torch
import torch.nn as nn

'''
# DeepNCM using Torch
https://github.com/mancinimassimiliano/DeepNCM
'''

class DeepNearestClassMean(nn.Module):
	
	# Initialize the classifier
	def __init__(self, features, classes, alpha=0.9):
		super(DeepNearestClassMean, self).__init__()
		self.means=nn.Parameter(torch.zeros(classes,features),requires_grad=False)				# Class Means
		self.running_means=nn.Parameter(torch.zeros(classes,features),requires_grad=False)
		self.alpha=alpha			# Mean decay value
		self.features=features			# Input features
		self.classes=classes
		

	# Forward pass (x=features)
	def forward(self,x):
		means_reshaped=self.means.view(1,self.classes,self.features).expand(x.shape[0],self.classes,self.features)
		features_reshaped=x.view(-1,1,self.features).expand(x.shape[0],self.classes,self.features)
		diff=(features_reshaped-means_reshaped)**2
		cumulative_diff=diff.sum(dim=-1)

		return -cumulative_diff
			
	
	# Update centers (x=features, y=labels)
	def update_means(self,x,y):
		for i in torch.unique(y):				# For each label
			N,mean=self.compute_mean(x,y,i)	# Compute mean

			# If labels already in the set, just update holder, otherwise add it to the model
			if N==0:
				self.running_means.data[i,:]=self.means.data[i,:]
			else:
				self.running_means.data[i,:]=mean
		
		# Update means
		self.update()
	

	# Perform the update following the mean decay procedure
	# Decay Mean / Eq.5
	def update(self):
		self.means.data=self.alpha*self.means.data+(1-self.alpha)*self.running_means


	# Compute mean by filtering the data of the same label
	def compute_mean(self,x,y,i):
		mask=(i==y).view(-1,1).float()
		mask=mask.cuda()
		N=mask.sum()
		if N==0:
			return N,0
		else:
			return N,(x.data*mask).sum(dim=0)/N