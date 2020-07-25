#=================================================================
#CS282 homework2 question3 PLA 
#Name: 効琦 ID: 2018232104
#=================================================================
import numpy as np 
import matplotlib.pyplot as plt 
import csv
import pandas as pd


class readcsv:
    def __init__(self,code='pla_a'):
        self.code=code
    
    def read(self):
        if self.code=='pla_a':
            X_train=pd.read_csv('G:\\myprojects\\pythonSpace\\hw2\\PLA_a\\train_data.csv')
            X_train=X_train.values[1:,:]
            y_train=pd.read_csv('G:\\myprojects\\pythonSpace\\hw2\\PLA_a\\train_label.csv')
            y_train=y_train.values[1:,:]
            y_train[y_train==0]=-1
            X_test=pd.read_csv('G:\\myprojects\\pythonSpace\\hw2\\PLA_a\\test_data.csv')
            X_test=X_test.values[1:,:]
            y_test=pd.read_csv('G:\\myprojects\\pythonSpace\\hw2\\PLA_a\\test_label.csv')
            y_test=y_test.values[1:,:]
            y_test[y_test==0]=-1
            return X_train, y_train, X_test, y_test
        else:
            X_trainb=pd.read_csv('G:\\myprojects\\pythonSpace\\hw2\\PLA_b\\train_data.csv')
            X_trainb=X_trainb.values[1:,:]
            y_trainb=pd.read_csv('G:\\myprojects\\pythonSpace\\hw2\\PLA_b\\train_label.csv')
            y_trainb=y_trainb.values[1:,:]
            y_trainb[y_trainb==0]=-1
            X_testb=pd.read_csv('G:\\myprojects\\pythonSpace\\hw2\\PLA_b\\test_data.csv')
            X_testb=X_testb.values[1:,:]
            y_testb=pd.read_csv('G:\\myprojects\\pythonSpace\\hw2\\PLA_b\\test_label.csv')
            y_testb=y_testb.values[1:,:]
            y_testb[y_testb==0]=-1
            return X_trainb, y_trainb, X_testb, y_testb
        '''
        if self.code=='pla_a':
            X_train=pd.read_csv('PLA_a\\train_data.csv')
            X_train=X_train.values[1:,:]
            y_train=pd.read_csv('PLA_a\\train_label.csv')
            y_train=y_train.values[1:,:]
            y_train[y_train==0]=-1
            X_test=pd.read_csv('PLA_a\\test_data.csv')
            X_test=X_test.values[1:,:]
            y_test=pd.read_csv('PLA_a\\test_label.csv')
            y_test=y_test.values[1:,:]
            y_test[y_test==0]=-1
            return X_train, y_train, X_test, y_test
        else:
            X_trainb=pd.read_csv('PLA_b\\train_data.csv')
            X_trainb=X_trainb.values[1:,:]
            y_trainb=pd.read_csv('PLA_b\\train_label.csv')
            y_trainb=y_trainb.values[1:,:]
            y_trainb[y_trainb==0]=-1
            X_testb=pd.read_csv('PLA_b\\test_data.csv')
            X_testb=X_testb.values[1:,:]
            y_testb=pd.read_csv('PLA_b\\test_label.csv')
            y_testb=y_testb.values[1:,:]
            y_testb[y_testb==0]=-1
        '''


class Perceptron:
    def __init__(self,x,label,a=1,mis_tolerate=0):
        self.x=x
        self.label=np.squeeze(label)
        #self.w=np.zeros((x.shape[1],1))#初始化权重，w1,w2均为0
        self.w=np.array([0,0])
        self.b=0
        self.a=1#学习率
        self.mis_tolerate=mis_tolerate
        self.numsamples=self.x.shape[0]
        self.numfeatures=self.x.shape[1]
        self.w3=np.array([0,0,0,0,0])
        self.x_new=np.array([0,0,0,0,0])
    def calcu(self,w,b,x):
        pre=np.dot(x,w)+b
        pre=int(np.squeeze(pre))
        pre=np.sign(pre)
        if pre == 0:
            pre==1
        return pre
    
    def calcu_3rd(self,w3,x):
        self.x_new=np.array([x[0]**3,x[0]**2,x[0],x[1],1])
        pre=np.dot(self.w3,self.x_new)
        pre=int(np.squeeze(pre))
        pre=np.sign(pre)
        if pre==0:
            pre=1
        return pre
    
    def update_3rd(self,label_i,data_i):
        deltaw3=(label_i*self.x_new).reshape(self.w3.shape)
        self.w3=self.w3+self.a*deltaw3

    def update(self,label_i,data_i):
        tmp=self.a*label_i*data_i
        tmp=tmp.reshape(self.w.shape)
        #更新w和b
        self.w=self.w+tmp
        self.b=self.b+self.a*label_i
 
    def train(self,p3rd='no'):
        flag=True
        epoch=0
        while flag:
            epoch=epoch+1
            count=0
            counts=[]
            predicts_train=[]
            for i in range(self.numsamples):
                if p3rd=='yes':
                    predict=self.calcu_3rd(self.w3,self.x[i,:])
                else:
                    predict=self.calcu(self.w,self.b,self.x[i,:])
                predicts_train.append(predict)
                if predict*self.label[i]<=0.0:
                    count+=1
                    if p3rd=='yes':
                        self.update_3rd(self.label[i],self.x[i,:])
                    else:
                        self.update(self.label[i],self.x[i,:])
            counts.append(count) 
            if count<=self.mis_tolerate:
                flag=False
            if epoch%100==0:
                print(epoch,self.w3,count) 
        if p3rd=='yes':
            return self.w3,predicts_train,counts
        else:
            return self.w,self.b,predicts_train,counts
    
    def pocket_tra(self,n_epoch=100000):
        einws=np.zeros(n_epoch)
        einw_1=0
        for j in range(n_epoch):
            einw=einw_1
            nerror=0
            wt=self.w
            bt=self.b
            for i in range(self.numsamples):
                predict=self.calcu(self.w,self.b,self.x[i,:])
                if predict*self.label[i]<=0.0:
                    nerror=nerror+1
                    self.update(self.label[i],self.x[i,:])
            einw_1=nerror/self.numsamples
            if einw_1 > einw:
                self.w=wt
                self.b=bt
            #ws.append(self.w)
            einws[j]=einw_1
            if j%1000==0:
                print(j,einws[j],einw)
        return einws,self.w,self.b

    def classifier(self,X_test,y_test,w,b,p3rd='no'):
        error=0
        error_rate=[]
        miscount=[]
        predicts_test=[]
        for i in range(X_test.shape[0]):
            if p3rd=='yes':
                pre=self.calcu_3rd(self.w3,X_test[i])
            else:
                pre=self.calcu(w,b,X_test[i])
            predicts_test.append(pre)
            if pre*y_test[i]<0.0:
                error=error+1
                miscount.append([X_test[i,:],y_test[i]])
            error_rate.append(error/(i+1))
        return error_rate,miscount,predicts_test
class view:
    def __init__(self,X_train,y_train,X_test,y_test,predicts_train,predicts_test,error_rate,w,b,count_rate,model='a'):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.predicts_train=predicts_train
        self.predicts_test=predicts_test
        self.w=w
        self.b=b
        self.error_rate=error_rate
        self.count_rate=count_rate
        self.model=model
    def fig(self):
        if self.model == 'a':
            plt.figure('Linearly separable training')
            plt.tight_layout()
        else:
            plt.figure('Linearly nonseparable training')
        plt.subplot(211)
        flagb=0
        flagg=0
        flagr=0
        for i in range(self.X_train.shape[0]):
            if (self.y_train[i]==-1) and (self.predicts_train[i]==-1):
                flagb=flagb+1
                if flagb==1:
                    plt.scatter(self.X_train[i,0],self.X_train[i,1],c='b',label='true negative',edgecolors='k')
                    plt.legend()
                else:
                    plt.scatter(self.X_train[i,0],self.X_train[i,1],c='b',edgecolors='k')               
            elif self.y_train[i]==1 and self.predicts_train[i]==1:
                flagg=flagg+1
                if flagg==1:
                    plt.scatter(self.X_train[i,0],self.X_train[i,1],c='g',label='true positive',edgecolors='k')
                    plt.legend()
                else:
                    plt.scatter(self.X_train[i,0],self.X_train[i,1],c='g',edgecolors='k')
            else :
                flagr=flagr+1
                if flagr==1:
                    plt.scatter(self.X_train[i,0],self.X_train[i,1],c='r',label='misclassify',edgecolors='k')
                    plt.legend()
                else:
                    plt.scatter(self.X_train[i,0],self.X_train[i,1],c='r',edgecolors='k')    
        xx=np.linspace(np.min(self.X_train[0])-20,np.max(self.X_train[0])+40,200)
        yy=(0.0-self.w[0]*xx+self.b)/(self.w[1])
        plt.plot(xx,yy,c='k',label='hyperplane')
        plt.legend()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("training set view")

        plt.subplot(212)
        plt.plot(self.count_rate,c='k')
        plt.xlabel("time")
        plt.ylabel('error_num')
        plt.title('trainning error curve')
        if self.model=='a':
            plt.figure('Linearly nseparable test')
        else:
            plt.figure('Linearly nonseparable')
        plt.subplot(211)
        flagb=0
        flagg=0
        flagr=0
        for i in range(self.X_test.shape[0]):
            if (self.y_test[i]==-1) and (self.predicts_test[i]==-1):
                flagb=flagb+1
                if flagb==1:
                    plt.scatter(self.X_test[i,0],self.X_test[i,1],c='b',label='true negative',edgecolors='k')
                    plt.legend()
                else:
                    plt.scatter(self.X_test[i,0],self.X_test[i,1],c='b',edgecolors='k')
            elif (self.y_test[i]==1) and (self.predicts_test[i]==1):
                flagg=flagg+1
                if flagg==1:
                    plt.scatter(self.X_test[i,0],self.X_test[i,1],c='g',label='true positive',edgecolors='k')
                    plt.legend()
                else:
                    plt.scatter(self.X_test[i,0],self.X_test[i,1],c='g',edgecolors='k')
            else:
                flagr=flagr+1
                if flagr==1:
                    plt.scatter(self.X_test[i,0],self.X_test[i,1],c='r',label='misclassify',edgecolors='k')
                    plt.legend()
                else:
                    plt.scatter(self.X_test[i,0],self.X_test[i,1],c='r',edgecolors='k')
        xx=np.linspace(np.min(self.X_test[0])-20,np.max(self.X_test[0])+40,200)
        yy=(0.0-self.w[0]*xx+self.b)/(self.w[1])
        plt.plot(xx,yy,c='k',label='hyperplane')
        plt.legend()
        plt.xlabel("testset x1 feature")
        plt.ylabel("testset x2 feature")
        plt.title("test dataset view")
        plt.subplot(212)
        plt.plot(self.error_rate,c='k')
        plt.xlabel("time")
        plt.ylabel("error_rate")
        plt.title("classify error")
        plt.show()

if __name__ == '__main__':

    #==============================PLA_a===================================
    
    myreadcvs=readcsv()
    X_train, y_train, X_test, y_test=myreadcvs.read()
    myperceptron=Perceptron(x=X_train,label=y_train)
    weights,bias,predicts_train,count_rate=myperceptron.train()
    error_rate,miscount,predicts_test=myperceptron.classifier(X_test=X_test,y_test=y_test,w=weights,b=bias)
    myview=view(X_train, y_train, X_test, y_test,predicts_train,predicts_test,error_rate,w=weights,b=bias,count_rate=count_rate)
    print("=====================PLA_a done=================")
    print("now please close PLA-a figure window for PLA_b running!!!")
    myview.fig()
    

    #=============================PLA_b=====================================
    myreadcvsb=readcsv(code='pla_b')
    X_trainb, y_trainb, X_testb, y_testb=myreadcvsb.read()
    myperceptronb=Perceptron(x=X_trainb,label=y_trainb,mis_tolerate=10)
    weightsb,biasb,predicts_trainb,count_rateb=myperceptronb.train()
    error_rateb,miscountb,predicts_testb=myperceptronb.classifier(X_test=X_testb,y_test=y_testb,w=weightsb,b=biasb)
    myviewb=view(X_train=X_trainb, y_train=y_trainb, X_test=X_testb, y_test=y_testb,predicts_train=predicts_trainb,predicts_test=predicts_testb,w=weightsb,b=biasb,error_rate=error_rateb,count_rate=count_rateb,model='b')
    print("======================PLA_b done======================")
    myviewb.fig()

    


    