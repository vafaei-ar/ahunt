from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pylab as plt
from functools import reduce
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,precision_score,classification_report,confusion_matrix,matthews_corrcoef

class Observetory:
    def __init__(self,plan,x,y):
        self.plan = plan
        self.x = x
        self.y = y
        self.x_obs = np.zeros( [0]+list(self.x.shape[1:]) )
        self.y_obs = np.zeros( [0]+list(self.y.shape[1:]) )
        self.num_obs = 0
        self.n_plan = len(self.plan)
#         self.new_obs()
    def new_obs(self,safemode=False,nightly=False):
        if self.num_obs==self.n_plan:
            if safemode:
#                 print('out of plan!')
                return self.x_obs,self.y_obs
            else:
                assert 0,'No more plan!' 
        data_config = self.plan[self.num_obs]
        self.x,self.y, x_obs, y_obs = data_prepare(self.x,self.y,data_config)
        self.x_obs = np.concatenate([self.x_obs,x_obs],axis=0)
        self.y_obs = np.concatenate([self.y_obs,y_obs],axis=0)
#         print(self.x_obs.shape, self.y_obs.shape, self.x.shape, self.y.shape)
        self.num_obs += 1
        if nightly:
            return x_obs,y_obs
        else:
            return self.x_obs,self.y_obs
    def delete(self,inds):
        self.x_obs = np.delete(self.x_obs,inds,axis=0)
        self.y_obs = np.delete(self.y_obs,inds,axis=0)

def describe_labels(y0,int_mapper=None,verbose=0):
    y = y0+0
    if y.ndim==2:
        y = np.argmax(y,axis=1)
    class_labels, nums = np.unique(y,return_counts=True)
    n_class = len(class_labels)
    tmp = '\t{:{width}s}/{:6d}\n'
    if verbose:
        print('labels/numbers are:')
        if int_mapper is None:
            ns = 6
    #         print('{:{width}s}/numbers are:'.format('labels',width=ns))
            print(*[tmp.format(str(i),j,width=ns) for i,j in zip(class_labels,nums)])
        else:
            ns = max([len(i) for i in list(int_mapper.values())])+3
    #         print('{:{width}s}/numbers are:'.format('labels',width=ns))
    #         print(tem.format(int_mapper[i],j,width=ns) for i,j in zip(class_labels,nums)])
            print(*[tmp.format(int_mapper[i],j,width=ns) for i,j in zip(class_labels,nums)])
    return n_class,class_labels, nums

# def describe_labels(y0):
#     y = y0+0
#     if y.ndim==2:
#         y = np.argmax(y,axis=1)
#     class_labels, nums = np.unique(y,return_counts=True)
#     n_calss = len(class_labels)
#     print('labels/numbers are:\n',*['{:5s}/{:6d}\n'.format(str(i),j) for i,j in zip(class_labels,nums)])
#     return n_calss

def shuffle_data(x,y):
    ndata = x.shape[0]
    inds = np.arange(ndata)
    np.random.shuffle(inds)
    return x[inds],y[inds]
    

def data_prepare(x,y,data_config,warning=True):
    if y.ndim>1:
        assert 0,'y dim problem!'
    dataset = []
    labels = []
    selects = []
    for k,v in data_config.items():
        filt = y==k
        if v>np.sum(filt):
            if warning:
                v = np.sum(filt)
                assert v!=0,'No data is available!'
                print('WARNING! Requested data is not available, reduce the {} class. The number is reduced to {}!'.format(k,v))
            else:
                assert 0, 'Requested data is not available, reduce the {} class'.format(k)
        
        inds = np.argwhere(filt)[:,0]
        np.random.shuffle(inds)
        selceted = inds[:v]
        dataset.extend(x[selceted])
        labels.extend([k]*v)
        selects.extend(selceted)
    return np.delete(x, selects,axis=0),np.delete(y, selects,axis=0),np.array(dataset),np.array(labels)

def load_npz(path,verbose=0):
    data = np.load(path)

    x = []
    y = []
    int_mapper = {}
    lbl_mapper = {}

    for i,key in enumerate(list(data)):
        dd = data[key]
        x.extend(dd)
        y.extend(dd.shape[0]*[i])
        int_mapper[i] = key
        lbl_mapper[key] = i
    x = np.array(x)
    y = np.array(y)

    if verbose:
        print(x.shape,y.shape)
        # describe_labels(y,int_mapper=None)
        describe_labels(y,int_mapper=int_mapper,verbose=verbose)

    return x,y,int_mapper,lbl_mapper

def plot_population(plan_tot,ax=None):
    epoches = np.arange(len(plan_tot))
    population_by_group = {}
    # for key in plan_tot[0].keys():
    keys = reduce(np.union1d, ([list(i.keys()) for i in plan_tot]))

    for key in keys:
        population_by_group[key] = []
    for i in plan_tot:
        nn = 0
        for key in keys:
            try:
                xx = i[key]
            except:
                xx = 0
            population_by_group[key].append(xx)
            nn += xx
        for key in keys:
            try:
                xx = i[key]
            except:
                xx = 0
            population_by_group[key][-1] = population_by_group[key][-1]/nn

#     population_by_group
    #         
    if ax is None:
        fig, ax = plt.subplots(figsize=(18,8))
        ax.set_xlim(0,len(plan_tot)-1)
        ax.set_ylim(0,1)
        ax.set_title('Class population',fontsize=14)
        ax.set_xlabel('epoch',fontsize=14)
        ax.set_ylabel('Number of data',fontsize=14)
        
    ax.stackplot(epoches, population_by_group.values(),
                 labels=population_by_group.keys(),)
    ax.legend(loc=(1,0.8))
        
    return population_by_group

def analyze(xx,cl=95):
    cl = 0.5*(100-cl)
    m = np.mean(xx,axis=0)
    l = np.percentile(xx,cl,axis=0)
    u = np.percentile(xx,100-cl,axis=0)
    return m,l,u

def analyze_plot(ax,metric,x=None,cl=95,clr='b',label='',alpha=0.3):
    m,l,u = analyze(metric,cl=cl)
    if x is None:
        x = np.arange(m.shape[0])
    ax.plot(x,m,clr,label=label)
    ax.fill_between(x,l,u,color=clr,alpha=alpha)

def rws_score2(outliers,v,n_o=None):
    outliers = np.array(outliers)
    if n_o is None:
        n_o = int(np.sum(outliers))
    b_s = np.arange(n_o)+1
    b_s = b_s[::-1]
    o_ind = np.argsort(v)[::-1]
    o_ind = o_ind[:n_o]
    return 1.*np.sum(b_s*outliers[o_ind].reshape(-1))/np.sum(b_s)

def rws_score(outliers,v,n_o=None):
    outliers = np.array(outliers)
    if n_o is None:
        n_o = int(np.sum(outliers))
    b_s = np.arange(n_o)+1
    o_ind = np.argsort(v)[-n_o:]
    return 1.*np.sum(b_s*outliers[o_ind].reshape(-1))/np.sum(b_s)

def check_int(x):
    return type(x) is int or 'int' in str(type(x))

def check_float(x):
    return type(x) is float or 'float' in str(type(x))

def get_tguess(n_questions,scr_ano,ano_inds):
    if check_int(n_questions):
        qinds = np.argsort(scr_ano)[-n_questions:]
    elif check_float(n_questions) and n_questions<1:
#             np.where(x<5)[0]
        qinds = np.where(scr_ano > n_questions)[0]
#             qinds = np.argsort(scr_ano)#[-:]
    else:
        print(type(n_questions))
        print(n_questions)
        assert 0,'Unknown number of questions.'

    inds = np.intersect1d(qinds,ano_inds)
    true_guess = len(inds)
    return inds,true_guess

class PredictionHistoryChecker:
    def __init__(self):
        self.seen = []

    def get_tguess(self,n_questions,scr_ano,ano_inds,x_obs):
        cands = np.argsort(scr_ano)[::-1]
        ncand = 0
        true_guess = 0
        for ic in cands:
            mn = 10000
            for asked in self.seen:
                dist = np.sum( (x_obs[ic]-asked)**2 )
                mn = min(mn,dist)
            if mn>1e-3:
                if ic in ano_inds:
                    true_guess = true_guess+1
                self.seen.append(x_obs[ic])
                ncand = ncand+1

            if check_int(n_questions):
                if ncand==n_questions: break
            elif check_float(n_questions) and n_questions<1:
                if scr_ano[ic] <  n_questions: break
            else:
                print(n_questions)
                assert 0,'Unknown number of questions.'
        return true_guess

# def rws_score(outliers,v,n_o=None):
#     outliers = np.array(outliers)
#     if n_o is None:
#         n_o = int(np.sum(outliers))
#     b_s = np.arange(n_o)+1
#     o_ind = np.argsort(v)[::-1]
#     o_ind = o_ind[:n_o]
#     return 1.*np.sum(b_s*outliers[o_ind].reshape(-1))/np.sum(b_s)
