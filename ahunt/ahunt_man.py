from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.keras.utils import to_categorical
from .models import VAE,build_model_2dconv,add_class
from .augment import balance_aug
from .data_utils import check_int,check_float


class AHunt:
    def __init__(self,x,y=None,interest=None,aug=None):
        self.x = x
        self.shape = self.x.shape
        
        if len(self.shape)==2:
            self.data_type = 'time series'
            assert 0, 'not suppoerted!'
        elif len(self.shape)==3:
            self.data_type = 'gray scale image'
            assert 0, 'not suppoerted!'
        elif len(self.shape)==4:
            self.data_type = 'colored image' 
        else:
            assert 0, 'Unknown data!'
        
#         print(self.data_type)
        
        self.n_data = self.shape[0]
        if y is None:
            self.rmove0 = False
            encoder,decoder,vae = VAE(shape = self.shape[1:],latent_dim = 2,l1 = 1e-10)
            H = vae.fit(x, x,
                        epochs=100,
                        batch_size=512,
                        verbose=0
                        )
            y = encoder.predict(x)[2]
            y = np.argmax(y,axis=1)
            self.lm = LabelManager(y)
            self.y = y
#             assert 0, 'not suppoerted!'
        else:
            self.rmove0 = False
#             n_class,class_labels, nums = describe_labels(y,int_mapper=None,verbose=1)
            self.lm = LabelManager(y)
            self.y = y
#         self.clf = clf
        self.n_class = self.lm.n_class+1
#         clf.layers[-1].output_shape[-1]
#         self.drt = drt

        self.clf,self.drt = build_model_2dconv(self.shape[1:],self.n_class,n_latent = 64)

        if interest is None:
            self.interest = self.n_class-1
        else:
            self.interest = self.lm.l2y[interest]
        self.aug = aug
        self.asked_q = []

    def fit(self,
            x=None,
            y=None,
            batch_size=256,
            epochs=10,
            validation_split=0.1,
            verbose=0,
            reshape=None
           ):
        if x is None or y is None:
            x = self.x
            y = self.y
        
#         print(x.shape,y.shape)

        if self.lm.update(y):
            self.n_class = self.lm.n_class+1
            self.clf = add_class(self.clf,self.drt,n_class = self.n_class,summary=0)

        yy = self.lm.to_y(y,onehot=1)
        xx,yy = balance_aug(x,yy,self.aug,reshape=reshape)
#         yy = to_categorical(yy, num_classes=self.n_class)
        yy = np.concatenate([yy,np.zeros((yy.shape[0],1))],axis=1)
    
        history = self.clf.fit(xx, yy,
                               batch_size=batch_size,
                               epochs=epochs,
                               validation_split=validation_split,
                               verbose=verbose)
        return history

    def ask_human(self,x,y,n_questions,minacc=0.0):
#         if self.lm.update(y):
#             print('model update in ask human!')
#             self.n_class = self.lm.n_class+1
#             self.clf = add_class(self.clf,self.drt,n_class = self.n_class,summary=1)
        
        
    
#         print('CHECK')
#         describe_labels(yy,verbose=1)
# #         self.clf.summary()
        
#         print(self.lm.n_class,self.n_class,np.unique(yy),self.interest)
#         print('END CHECK')
        if check_int(n_questions):
            yy = self.lm.to_y(y,onehot=0,add_new=1)
            out_obs = yy==self.interest
            ano_inds = np.argwhere(out_obs)[:,0]
            z_clf = self.clf.predict(x)
            scr_ano = z_clf[:,self.interest]
            qlist = np.argsort(scr_ano)[::-1]
            inds_all = []
            inds_interest = []
            for q in qlist:
                mn = 10000
                for asked in self.asked_q:
                    dist = np.sum( (x[q]-asked)**2 )
                    mn = min(mn,dist)
                if mn>minacc:
                    inds_all.append(q)
                    if q in ano_inds:
                        inds_interest.append(q)
                if len(inds_all)==n_questions: break
            return inds_all,inds_interest

        elif check_float(n_questions) and n_questions<1:
            yy = self.lm.to_y(y,onehot=0,add_new=1)
            out_obs = yy==self.interest
            ano_inds = np.argwhere(out_obs)[:,0]
            z_clf = self.clf.predict(x)
            scr_ano = z_clf[:,self.interest]
            qlist = np.argsort(scr_ano)[::-1]
            inds_all = []
            inds_interest = []
            for q in qlist:
                mn = 10000
                for asked in self.asked_q:
                    dist = np.sum( (x[q]-asked)**2 )
                    mn = min(mn,dist)
                if mn>minacc:
                    inds_all.append(q)
                    if q in ano_inds:
                        inds_interest.append(q)
#                 print(scr_ano[q],  n_questions)
                if scr_ano[q] <  n_questions: break
            
#             print('Number of questions is {}.'.format(len(inds_all)))
            return inds_all,inds_interest

        else:
            assert 0,'Unknown number of questions.'

        

    
#     def human_call1(self,x,y,n_questions,minacc=0.0):
# #         ano_inds = np.argwhere(out2)[:,0]

#         inds_all,inds_interest = self.ask_human(x,y,n_questions,minacc=minacc)
# #         [-n_questions:]
# #         inds = np.intersect1d(qinds,ano_inds)
#         true_guess = len(inds_interest)
#         self.asked_q.extend(x[inds_interest])
#         self.x = np.concatenate([x[inds_interest],self.x],axis=0)
#         self.n_data = self.x.shape[0]
#         self.y = np.concatenate([y[inds_interest],self.y],axis=0)
#         return true_guess

    def human_call(self,x,y,n_questions,minacc=0.0):
#         ano_inds = np.argwhere(out2)[:,0]
#         scr_ano = self.predict(x)
#         qinds = np.argsort(scr_ano)[-n_questions:]
#         inds = np.intersect1d(qinds,ano_inds)
        inds_all,inds_interest = self.ask_human(x,y,n_questions,minacc=minacc)
        true_guess = len(inds_interest)
        self.asked_q.extend(x[inds_all])
        if self.rmove0:
            self.x = x[inds_all]
            self.y = y[inds_all]
        else:
            self.x = np.concatenate([x[inds_all],self.x],axis=0)
            self.y = np.concatenate([y[inds_all],self.y],axis=0)
        self.n_data = self.x.shape[0]
        
        return true_guess

    def predict(self,x):
        z_clf = self.clf.predict(x)
        return z_clf[:,self.interest]
    
    def class_predict(self,x):
        return self.clf.predict(x)
    
    def to_latent(self,x):
        z_mu = self.drt.predict(x)
        return z_mu

class LabelManager():
    def __init__(self,labels):
        labelsp = np.array(labels).astype(np.int).astype(np.str)
        assert labelsp.ndim==1,'the label array has to be 1D.' 
        self.yints = np.unique(labelsp)
        self.l2y = {j:i for i,j in enumerate(self.yints)}
        self.y2l = {i:j for i,j in enumerate(self.yints)}
        self.n_class = len(self.yints)
                        
    def to_labels(self,y):
        yp = np.array(y)
        if yp.ndim==2:
            yp = np.argmax(yp,axis=1)
        labels = []
        for i in yp:
            labels.append(self.y2l[i])
        labels = np.array(labels)
        return labels
        
    def to_y(self,labels,onehot=False,add_new=False):
        y = []
        for i in np.array(labels).astype(np.int).astype(np.str):
            if add_new:
                if i in self.l2y.keys():
                    y.append(self.l2y[i])
                else:
                    y.append(self.n_class)
            else:
                y.append(self.l2y[i])
        y = np.array(y)
        
        if onehot:
            return to_categorical(y)
        return y

    def __call__(self,labels,onehot=False,add_new=False):
        return self.to_y(labels,onehot=onehot,add_new=add_new)
    
    def update(self,labels):
        yints = np.unique(labels).astype(np.int).astype(np.str)
        dff = np.setdiff1d(yints,self.yints)
        if len(dff)==0:
#             print('No change!')
            return False
        l2y = {j:i+self.n_class for i,j in enumerate(dff)}
        y2l = {i+self.n_class:j for i,j in enumerate(dff)}
        
        self.l2y = {**self.l2y, **l2y}
        self.y2l = {**self.y2l, **y2l}
        self.yints = np.union1d(self.yints,dff)
        self.n_class = len(self.yints)
        return True



# class AHunt:
#     def __init__(self,x,y,clf,drt,interest,aug=None):
#         self.x = x
#         self.n_data = self.x.shape[0]
#         self.y = y
#         self.clf = clf
#         self.n_class = clf.layers[-1].output_shape[-1]
#         self.drt = drt
#         self.interest = interest
#         self.aug = aug
#         self.asked_q = []

#     def fit(self,
#             x=None,
#             y=None,
#             batch_size=256,
#             epochs=10,
#             validation_split=0.1,
#             verbose=0,
#             reshape=None
#            ):
#         if x is None or y is None:
#             x = self.x
#             y = self.y
        
# #         print(x.shape,y.shape)
#         xx,yy = balance_aug(x,y,self.aug,reshape=reshape)
#         yy = to_categorical(yy, num_classes=self.n_class)
#         history = self.clf.fit(xx, yy,
#                                batch_size=batch_size,
#                                epochs=epochs,
#                                validation_split=validation_split,
#                                verbose=verbose)
#         return history

#     def ask_human(self,x,y,n_questions,minacc=0.0):
#         out_obs = y==self.interest
#         ano_inds = np.argwhere(out_obs)[:,0]
#         z_clf = self.clf.predict(x)
#         scr_ano = z_clf[:,self.interest]
#         qlist = np.argsort(scr_ano)[::-1]
#         inds_all = []
#         inds_interest = []
#         for q in qlist:
#             mn = 10000
#             for asked in self.asked_q:
#                 dist = np.sum( (x[q]-asked)**2 )
#                 mn = min(mn,dist)
#             if mn>minacc:
#                 inds_all.append(q)
#                 if q in ano_inds:
#                     inds_interest.append(q)
#             if len(inds_all)==n_questions: break
#         return inds_all,inds_interest
    
#     def human_call1(self,x,y,n_questions,minacc=0.0):
# #         ano_inds = np.argwhere(out2)[:,0]

#         inds_all,inds_interest = self.ask_human(x,y,n_questions,minacc=minacc)
# #         [-n_questions:]
# #         inds = np.intersect1d(qinds,ano_inds)
#         true_guess = len(inds_interest)
#         self.asked_q.extend(x[inds_interest])
#         self.x = np.concatenate([x[inds_interest],self.x],axis=0)
#         self.n_data = self.x.shape[0]
#         self.y = np.concatenate([y[inds_interest],self.y],axis=0)
#         return true_guess

#     def human_call2(self,x,y,n_questions,minacc=0.0):
# #         ano_inds = np.argwhere(out2)[:,0]
# #         scr_ano = self.predict(x)
# #         qinds = np.argsort(scr_ano)[-n_questions:]
# #         inds = np.intersect1d(qinds,ano_inds)
#         inds_all,inds_interest = self.ask_human(x,y,n_questions,minacc=minacc)
#         true_guess = len(inds_interest)
#         self.asked_q.extend(x[inds_all])
#         self.x = np.concatenate([x[inds_all],self.x],axis=0)
#         self.n_data = self.x.shape[0]
#         self.y = np.concatenate([y[inds_all],self.y],axis=0)
#         return true_guess

#     def predict(self,x):
#         z_clf = self.clf.predict(x)
#         return z_clf[:,self.interest]
    
#     def class_predict(self,x):
#         return self.clf.predict(x)
    
#     def to_latent(self,x):
#         z_mu = self.drt.predict(x)
#         return z_mu


