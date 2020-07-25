# -*- coding:utf-8 -*- 
import numpy as np 
import matplotlib.pyplot as plt 
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from numpy.random import normal,random,uniform
import datetime
import sys
sys.path.append(r"G:\myprojects\pythonSpace\hw2")
from pla import Perceptron



class semi_data_generator:
    def __init__(self,rad=10,thk=5,sep=5,nsamples=2000):
        self.rad=rad
        self.thk=thk
        self.sep=sep
        self.nsamples=nsamples
    def gener(self):
        tflag=True
        #center=np.array([0,0])
        tpoints=[]
        while tflag:
            tpoint=np.array([np.random.uniform(-15,15),np.random.uniform(0,15)],dtype='float16')
            dist2=tpoint[0]**2+tpoint[1]**2
            if (dist2 >= 100) and (dist2 <=225):
                tpoints.append(tpoint)
            if len(tpoints) >= self.nsamples/2:
                tflag=False
        tpoints=np.array(tpoints)
        bflag=True        
        bpoints=[]
        while bflag:
            bpoint=np.array([np.random.uniform(-15,15),np.random.uniform(-15,0)],dtype='float16')
            dist2=bpoint[0]**2+bpoint[1]**2
            if (dist2 >= 100) and (dist2 <=225):
                bpoints.append(bpoint)
            if len(bpoints) >= self.nsamples/2:
                bflag=False
        bpoints=np.array(bpoints)
        shift=np.array([self.thk/2+self.rad,-self.sep])
        bpoints=bpoints+shift
        return tpoints,bpoints

    def makedata(self,tpoints,bpoints):
        
        X=np.vstack((tpoints,bpoints))
        y=np.ones(self.nsamples)
        y[int(self.nsamples/2):]=-1
        y_copy=y.copy()
        X_copy=X.copy()
        index=np.random.permutation(range(self.nsamples))
        for i in range(X.shape[0]):
            X[i,:]=X_copy[index[i],:]
            y[i]=y_copy[index[i]]
        np.savetxt('semiX_train.csv',X,delimiter=',')
        np.savetxt('semiy_train.csv',y,delimiter=',')
        return X,y

if __name__=='__main__':
    
    mydata=semi_data_generator()
    tpoints,bpoints=mydata.gener()
    X,y=mydata.makedata(tpoints,bpoints)

    plt.figure('PLA versus LinearRegression')
    plt.scatter(tpoints[:,0],tpoints[:,1],c='r',label='positive')
    plt.legend()
    plt.scatter(bpoints[:,0],bpoints[:,1],c='b',label='negative')
    plt.legend()
    
    #===========PLA===============================
    mypla=Perceptron(x=X,label=y)
    weights,bias,predicts_train,count_rate=mypla.train()
    pla_x=np.linspace(np.min(tpoints[:,0]),np.max(bpoints[:,0]),300)
    pla_y=-(weights[0]*pla_x+bias)/weights[1]
    plt.plot(pla_x,pla_y,c='g',label='pla')
    plt.legend()
    #========LinearRegression======
    mylineg=LinearRegression()
    mylineg.fit(X,y)
    k,b=mylineg.coef_[0],mylineg.intercept_
    lig_x=np.linspace(np.min(tpoints[:,0]),np.max(bpoints[:,0]),300)
    lig_y=k*lig_x+b
    plt.plot(lig_x,lig_y,c='k',label='LinearRegression')
    plt.legend()

    #========================================================b========================================
    seps=np.linspace(0.2,5,25)
    mydatabs=[]
    times=[]
    for i in range(25):
        mydatab=semi_data_generator(sep=seps[i])
        mydatabs.append(mydatab)
        tpoints,bpoints=mydatab.gener()
        X,y=mydata.makedata(tpoints,bpoints)

        start=datetime.datetime.now()
        myplab=Perceptron(x=X,label=y)
        weights,bias,predicts_train,count_rate=myplab.train()
        end=datetime.datetime.now()
        time=(end-start).total_seconds()
        times.append(time)
    plt.figure('problem b Convergent time versus sep value')
    plt.plot(seps,times,c='r',marker='o',label='Iterative time')
    plt.xticks(seps)
    plt.legend()
    plt.xlabel("Sep value")
    plt.ylabel("Fit time /s")
    plt.title('sep-time viewer')
    
    #=================================================c=================================================
    # [Solution] when sep=-5, obviously classical PLA won't get convergent state. Because a lot of nonseparable points
    # lead to index of error_count won't be zero . The algorithm will continue to circulate.
    sepc=-5
    n_epoch=1000
    mydatac=semi_data_generator(sep=sepc)
    tpointsc,bpointsc=mydatac.gener()
    X,y=mydatac.makedata(tpointsc,bpointsc)
    
    myplac=Perceptron(x=X,label=y)
    # use pocket algorithm train sep=-5 data model
    einws,weightsc,biasc=myplac.pocket_tra(n_epoch=n_epoch)
    plt.figure('pocket algorithm')
    plt.subplot(211)
    plt.scatter(tpointsc[:,0],tpointsc[:,1],c='r',edgecolors='k',label='positive')
    plt.legend()
    plt.scatter(bpointsc[:,0],bpointsc[:,1],c='b',edgecolors='k',label='negative')
    plt.legend()
    pocket_x=np.linspace(np.min(tpointsc[:,0]),np.max(bpointsc[:,0]),300)
    pocket_y=-(pocket_x*weightsc[0]+biasc)/(weightsc[1]+0.001)
    plt.plot(pocket_x,pocket_y,c='k',label='pocket hyperplane')
    plt.legend()
    plt.title('pocket hypothesis curve')

    plt.subplot(212)
    # sampling the so long 1D-array einws for better plotting
    einws_sam=[]
    for i in range(einws.shape[0]):
        if i%10==0:
            einws_sam.append(einws[i])
    plt.plot(einws_sam,c='k')
    plt.xlabel('n-epoch:100000')
    plt.ylabel('Ein(w)')
    plt.title('Ein(w)-n_epoch curve')
    
#=============================================d classifier for linearly nonseparable ==================================
    
    #================================ nonlinear-3rd poly polynomial=======================
    myp3rd=Perceptron(x=X,label=y,mis_tolerate=30)
    weights3,predicts_train,counts=myp3rd.train(p3rd='yes')
    plt.figure('nonlinear-3rd poly polynomial ')
    #plt.subplot(211)

    plt.scatter(tpointsc[:,0],tpointsc[:,1],c='r',edgecolors='k',label='positive')
    plt.legend()
    plt.scatter(bpointsc[:,0],bpointsc[:,1],c='b',edgecolors='k',label='negative')
    plt.legend()
    p3rd_x=np.linspace(-10,20,300)
    p3rd_y=-(weights3[0]*(p3rd_x**3)+weights3[1]*(p3rd_x**2)+weights3[2]*p3rd_x+weights3[4])/weights3[3]
    
    # 错分样本数为25时，获得的w3=[ -3354.16942695    62485.27819845     2233.99629974   413355.09189606      -2229007.] 带入 可以验证拟合的曲线
    # 3阶多项式拟合极慢 ，可通过我运行过获得的值快速的检验代码的正确性！
    # p3rd_y=-(-3354.16942695*(p3rd_x**3)+ 62485.27819845*(p3rd_x**2)+ 2233.99629974 *p3rd_x+ -2229007)/413355.09189606 
    
    plt.plot(p3rd_x,p3rd_y,c='k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('3rd-fit')
    
    #plt.subplot(212)
    plt.figure('n_error-epoch')
    counts_sam=[]
    for i in range(len(counts)):
        counts_sam.append(counts[100*i])
    
    plt.plot(counts_sam,c='r')
    plt.xlabel('itera times')
    plt.ylabel('Ein')
    plt.title('Ein-itears curve')
    
    plt.show()
                
        