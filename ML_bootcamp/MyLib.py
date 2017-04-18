

class Data:
    pd = __import__('pandas')
    np = __import__('numpy')
    sk = __import__('sklearn')
    import matplotlib.pyplot as plt

    #hist = importlib.import_module('matplotlib.pyplot.hist')
    #scatter = importlib.import_module('matplotlib.pyplot.scatter')
    def __init__(self,X,y):

        if (type(X) is type(self.pd.DataFrame())):
            self.df_X = X
            self.df_y = y
            self.keys= X.keys()
            self.df = True
        else:
            self.df = False
        self.scaled  = False
        self.X=self.np.asarray(X)
        self.y=self.np.asarray(y).reshape(len(y),)
        
	#TODO:if changing x direx=ctly from log shape not updates
	self.shape =  self.X.shape,self.y.shape

    def get_train_test(self,ratio=0.2):
        x_tr,x_t,y_tr,y_t = self.sk.model_selection.train_test_split(self.X,self.y,test_size = ratio)
        return (x_tr,y_tr),(x_t,y_t)
    
    def log_x(self,fechs = [0,1,3,4,6,8,9,10,11]):
        X_lg = []
        loged  = Data(self.X,self.y)
        X = self.X
        for i in fechs:
            X_lg.append(self.np.log(X[:,i]+1))
        X_lg = self.np.asarray(X_lg)
        loged.X = X_lg.T
        loged.df = False
        loged.loged = True
        return loged
    def shuffle(self):
        Xs,ys=self.sk.utils.shuffle(self.X,self.y)
        self.X = Xs
        self.y = ys
        
    def scale(self):
        if (self.scaled):
            print 'Already scaled'
        else:
            pr = __import__('sklearn.preprocessing')
            scaler = pr.preprocessing.StandardScaler()
            X =  self.X
            scaler.fit(X)
            self.scaler = scaler
            tr = scaler.transform(X)
            self.orig_X = X
            self.X = tr
            self.scaled = True
            self.df = False
            return tr
    
    def describe(self):
        if self.df:
            return self.df_X.describe()
        else:
            return self.pd.DataFrame(self.X).describe()
        
    def plot_hist(self,num=0,bins=30,bw=0.4,kde=True,x=None):
	
	w = self.X[:,num]
	if kde:
		from scipy.stats.kde import gaussian_kde
		KDEpdf0 = gaussian_kde(w[self.y==0],bw_method=bw)
		KDEpdf1 = gaussian_kde(w[self.y==1],bw_method=bw)
		
		self.plt.plot(x,KDEpdf0(x),'g--',alpha=0.8)
		self.plt.plot(x,KDEpdf1(x),'b--',alpha=0.8)

        self.plt.hist(w[self.y==0], bins=bins, alpha=0.65, normed=True, label='notRet')
	
        self.plt.hist(w[self.y==1], bins=bins, alpha=0.65, normed=True, label='ret')
        self.plt.legend(loc='upper right')
        try:
            print 'plot for', self.keys[num]
        except:
            print ""
    def plot_scatter(self,a=0,b=1,alpha= 0.5):
        x= self.X[:,a]
        y = self.X[:,b]
        self.scatter(x,y,s=5,c=self.y,cmap='RdYlGn',alpha=alpha,edgecolors='face')
        
        
    def pairplot(self,l = 0,kws={"s": 7}):
        seaborn = __import__('seaborn')
        if(self.df):
            if(l==0):
                l= self.df_X.shape()[0]
            pair = pd.DataFrame(self.df_X[:l])
            pair['labels'] = self.df_y[:l]
            seaborn.pairplot(pair[:l], hue = 'labels',plot_kws=kws)
        else:
            if(l==0):
                l= self.X.shape()[0]
            pair = pd.DataFrame(self.X[:l])
            pair['labels'] = self.y[:l]
            seaborn.pairplot(pair[:l], hue = 'labels',plot_kws=kws)
