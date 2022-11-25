import numpy as np
from matplotlib.patches import Ellipse


#Compute the distance between two points
def distance(a,b):
	return(np.sqrt(np.sum((a-b)**2,axis=0)))


def init_barycenter(Data,K):  #K : nb of clusters
	d, n = Data.shape[0], Data.shape[1]
	Centers = np.zeros((K,d))  #will contain our barycenters
	a = np.random.randint(n)  # we randomly choose a point in our data set. This point is our first barycenter
	Centers[0,:] = Data[:,a]
	mask = np.ones(n, dtype=bool)
	mask[a]= False
	B = Data[:,mask]  #B is the data points matrix without the barycenters
	for k in range(K-1):
		nk = B.shape[1]
		Distances = np.zeros((k+1,nk))  #size (nb de centre)x(n-nb de centre) : distances between centers and all data points
		for i in range(k+1):
			A = np.reshape(Centers[i,:],(d,1))
			dist = distance(A,B)
			Distances[i,:] = dist
		Proba = np.min(Distances,axis=0)
		Proba *= Proba
		Proba = Proba/np.sum(Proba)
		c = np.random.choice(np.arange(nk),p = Proba)  #The new barycenter is chosen with a probability proportional to the minimum distance to the centers.
		Centers[k+1,:]=B[:,c]
		mask = np.ones(nk, dtype=bool)
		mask[c]= False
		B = B[:,mask]  #We update B
	return Centers


#Kmeans ++
def Kmeans(Data,K,IterMax = 1000):  #K : nb of clusters
	d, n = Data.shape[0], Data.shape[1]
	Centers = init_barycenter(Data,K)
	Tau = np.zeros((K,n))
	Sum = []
	for t in range(IterMax):
		s = 0
		New_Tau = np.zeros((K,n))
		#gives Tau with the actual centers
		for i in range(n):
			A = np.reshape(Data[:,i],(d,1))
			B = Centers.T
			dist = distance(A,B)
			ind = int(np.argmin(dist))
			New_Tau[ind,i]=1
			s += np.min(dist**2)

		#gives the new centers with the actual Tau
		for k in range(K):
			mask = np.array(New_Tau[k,:],dtype=bool)
			Clust = Data[:,mask]
			Nk = np.sum(New_Tau[k,:])
			Bary = np.sum(Clust,axis=1)
			Centers[k,:] = Bary/Nk
		
		Sum.append(s)
		#Stop criterion
		if np.all((Tau - New_Tau) == 0) : break  #we stop when there is no change in the Tau matrix
		Tau = np.copy(New_Tau)
	#print('Number of iterations in kmeans : ',t)
		
	return Tau,Sum


#Function that compute Tau for pi, mu and sigma given
def E_step(Pi,Mu,Sigma,X):
    d = X.shape[0]
    n = X.shape[1]
    K = Mu.shape[0]
    Tau = np.empty((K,n))
    for k in range(K):
        Mu_k = np.reshape(Mu[k,:],(d,1))
        Sigma_k = np.reshape(Sigma[k,:,:],(d,d))
        #At some point there is a negative eigenvalue very close to zero and it causes an overflow in the exponential. 
        # We avoid this by adding the identity matrix multiplied by 10e-10 to the matrix Sigma
        Sigma_k += 1e-5*np.eye(d)
        Sigma_inv = np.linalg.inv(Sigma_k)
        Sigma_det = np.linalg.det(Sigma_k)
        g = -.5 * np.einsum("xi, xy, yi -> i", X-Mu_k, Sigma_inv, X-Mu_k)
        Tau[k] = Pi[k]*np.exp(g)/(Sigma_det*(2*np.pi)**d)**(1/2)
    L = np.sum(Tau, 0)
    log_like = np.log(L).sum()
    Tau /= L
    return Tau,log_like


#Function that compute pi, mu and sigma for a given Tau
def M_step(tau,data, IS = False, W = 1):
	K = tau.shape[0]
	n = tau.shape[1]
	d = data.shape[0]
	#Pi
	if IS == True:
		W = np.reshape(W,(1,n))
		tau = tau*W
	Pi = np.sum(tau, axis=1)  #size K
	Pi /= np.sum(Pi)
	Pi = np.reshape(Pi,(K,1))
	
	#Mu
	N = np.sum(tau,axis=1)   #size K
	Mu = np.dot(tau,data.T)/np.reshape(N,(K,1)) #size kxd

	#Sigma
	Sigma = np.zeros((K,d,d))
	for k in range(K):
		Mu_k = np.reshape(Mu[k,:],(d,1))
		tau_k = np.reshape(tau[k,:],(1,n))
		s = np.dot((data - Mu_k)*tau_k,(data-Mu_k).T)/N[k]
		Sigma[k,:,:] = s

	return Pi,Mu,Sigma


def EM(Data,K,IterMax=2000, IS = False,W = 1):
	#Data shape : n x d
	#K : nb of cluster
	Tau,_ = Kmeans(Data,K)
	Log_like = [0]
	for t in range(IterMax):
		Pi,Mu,Sigma = M_step(Tau,Data,IS = IS, W = W)              #M-step
		Tau,log_like = E_step(Pi,Mu,Sigma,Data)     #E-step

		Log_like.append(log_like)
		#Stop criterion
		if np.abs(log_like-Log_like[t]) < 10e-3: break
	
	d = {
         "Pi": Pi, 
         "Mu" : Mu, 
         "Sigma" : Sigma,
		 "Log-likelihood" : Log_like[1:]}
	
	#print('Number of iterations in EM : ',t)
	return d


#### Plot EM graph ####

def plot_graph_EM(Data,mu,Sigma,pi,ax):
    ax.scatter(mu[:,0], mu[:,1], c='black', s=100*pi)
    ax.scatter(Data[0,:], Data[1,:], c='gold', alpha=.7)
    for i in range(len(mu)):
        w, v = np.linalg.eigh(Sigma[i])
        w = .3 * w**.5
        angle = np.math.atan2(v[0,1], v[0,0])*180/np.pi
        for mul in [4, 6, 10]:
            ax.add_patch(Ellipse(mu[i], mul*w[0], mul*w[1], angle,
                            facecolor='none', edgecolor='black', linestyle='--', linewidth=.4+.4*mul*pi[i]))


##### Sampling multivariate gaussian distribution #####

def Gaussian_mixture_law(X,Mu,Sigma,Alpha): 
    m,d = Mu.shape[0],Mu.shape[1]
    n = X.shape[1]
    Proba = np.zeros(n)
    for j in range(m):
        Mu_j = np.reshape(Mu[j,:],(d,1))
        Sigma_j = np.reshape(Sigma[j,:,:],(d,d))
        Sigma_inv = np.linalg.inv(Sigma_j)
        Sigma_det = np.linalg.det(Sigma_j)
        g = -.5 * np.einsum("xi, xy, yi -> i", X-Mu_j, Sigma_inv, X-Mu_j)
        Proba += Alpha[j]*np.exp(g)/(Sigma_det*(2*np.pi)**d)**(1/2)
    return Proba

def inv_cumulative_fct(P,x):
	sum_p = 0
	n = len(P)
	for i in range(n):
		sum_p += P[i]
		if sum_p >=x:
			return i

def random_var(X,P,N):
	u = np.random.rand(N)
	d = X.shape[0]
	Var = np.zeros((d,N))
	for k in range(len(u)):
		j = inv_cumulative_fct(P,u[k])
		Var[:,k]=X[:,j]
	return Var