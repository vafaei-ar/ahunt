from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.keras.utils import to_categorical
from .models import VAE,build_model_2dconv,add_class
from .augment import balance_aug
from .data_utils import check_int,check_float


class AHunt:
    def __init__(self,x,y=None,interest=None,aug=None,reserved_classes=['r1']):
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
            self.lm = LabelManager(y,reserved_classes=reserved_classes)
            self.y = y
#         self.clf = clf
#         self.n_reserve = n_reserve
        self.n_class = self.lm.n_class#+self.n_reserve
#         clf.layers[-1].output_shape[-1]
#         self.drt = drt

        self.clf,self.drt = build_model_2dconv(self.shape[1:],self.n_class,n_latent = 64)

#         if interest is None:
#             self.interest = self.n_class-1
#         else:
#             self.interest = self.lm.l2y[interest]
        self.set_interest(interest)
        self.aug = aug
        self.asked_q = []
        self.nqs = {}
        self.maxlimq = 100

    def set_interest(self,interest):
        if interest is None:
            if 'r1' in self.lm.l2y.keys():
                self.interest = {self.lm.l2y['r1']:1}
            else:
                print('Interest is not set while you did not set any reserved class. {} class is assumed as interest.'.format(self.lm.y2l[0]))
                self.interest = {0:1}
        elif type(interest) is int or type(interest) is str:
            assert interest in self.lm.l2y.keys(), 'Requested lable is not in {}'.format(self.lm.l2y.keys())
            self.interest = {self.lm.l2y[interest]:1}
        else:
            assert type(interest) is dict, 'interest is not dictionary. Example {"class1":0.1,"class2":0.9}'
            self.interest = {self.lm.l2y[i]:w for i,w in interest.items()}

    def get_interest(self):
        return {self.lm.y2l[i]:w for i,w in self.interest.items()}
            
    def fit(self,
            x=None,
            y=None,
            batch_size=256,
            epochs=10,
            validation_split=0.1,
            verbose=0,
            reshape=None,
            ivc=5, # intra-variability coefficient
            wmax=2 # IVC w_max
           ):
        if x is None or y is None:
            x = self.x
            y = self.y
        
#         print(x.shape,y.shape)

        if self.lm.update(y):
            self.n_class = self.lm.n_class#+self.n_reserve
            self.clf = add_class(self.clf,self.drt,n_class = self.n_class,summary=0)

#         yy = self.lm.to_y(y,onehot=1)
#         xx,yy = balance_aug(x,yy,self.aug,reshape=reshape)
# #         yy = to_categorical(yy, num_classes=self.n_class)
#         yy = np.concatenate([yy,np.zeros((yy.shape[0],self.n_reserve))],axis=1)
        def tocater(y):
            return self.lm.to_y(y,onehot=1)
        xx,yy = balance_aug(x,y,self.aug,reshape=reshape,tocater=tocater)
  
        if ivc==0:
            history = self.clf.fit(self.aug.flow(xx, yy, batch_size=batch_size),
                                   steps_per_epoch=len(xx) // batch_size,
                                   epochs=epochs,
                                   verbose=verbose)

#             history = self.clf.fit(xx, yy,
#                                    batch_size=batch_size,
#                                    epochs=epochs,
#                                    validation_split=validation_split,
#                                    verbose=verbose)
            return history.history
        else:
            dc = DataContainer(xx, yy)
            xc, yc = dc.process(self.clf,c=ivc,wmax=wmax)
            histories = []
            for i in range(epochs):
                history = self.clf.fit(self.aug.flow(xc, yc, batch_size=batch_size),
                                       steps_per_epoch=len(xc) // batch_size,
                                       epochs=1,
                                       verbose=verbose)
#                 history = self.clf.fit(xc, yc,
#                                        batch_size=batch_size,
#                                        epochs=1,
#                                        validation_split=validation_split,
#                                        verbose=verbose

                xc, yc = dc.process(self.clf,c=ivc,wmax=wmax)
                histories.append(history)
            del dc,xx,yy
                
            if 'val_accuracy' in history.history.keys():
                hist = {'accuracy':[history.history['accuracy'] for history in histories],
                        'val_accuracy':[history.history['val_accuracy'] for history in histories]}
            else:
                hist = {'accuracy':[history.history['accuracy'] for history in histories]}
            return hist

#     def ask_human(self,x,y,n_questions,minacc=0.0):
# #         if self.lm.update(y):
# #             print('model update in ask human!')
# #             self.n_class = self.lm.n_class+1
# #             self.clf = add_class(self.clf,self.drt,n_class = self.n_class,summary=1)
        
        
    
# #         print('CHECK')
# #         describe_labels(yy,verbose=1)
# # #         self.clf.summary()
        
# #         print(self.lm.n_class,self.n_class,np.unique(yy),self.interest)
# #         print('END CHECK')
#         if check_int(n_questions):
#             yy = self.lm.to_y(y,onehot=0,add_new=1)
#             out_obs = yy==self.interest
#             ano_inds = np.argwhere(out_obs)[:,0]
#             z_clf = self.clf.predict(x)
#             scr_ano = z_clf[:,self.interest]
#             qlist = np.argsort(scr_ano)[::-1]
#             inds_all = []
#             inds_interest = []
#             for q in qlist:
#                 mn = 10000
#                 for asked in self.asked_q:
#                     dist = np.sum( (x[q]-asked)**2 )
#                     mn = min(mn,dist)
#                 if mn>minacc:
#                     inds_all.append(q)
#                     if q in ano_inds:
#                         inds_interest.append(q)
#                 if len(inds_all)==n_questions: break
#             return inds_all,inds_interest

#         elif check_float(n_questions) and n_questions<1:
#             yy = self.lm.to_y(y,onehot=0,add_new=1)
#             out_obs = yy==self.interest
#             ano_inds = np.argwhere(out_obs)[:,0]
#             z_clf = self.clf.predict(x)
#             scr_ano = z_clf[:,self.interest]
#             qlist = np.argsort(scr_ano)[::-1]
#             inds_all = []
#             inds_interest = []
#             for q in qlist:
#                 mn = 10000
#                 for asked in self.asked_q:
#                     dist = np.sum( (x[q]-asked)**2 )
#                     mn = min(mn,dist)
#                 if mn>minacc:
#                     inds_all.append(q)
#                     if q in ano_inds:
#                         inds_interest.append(q)
# #                 print(scr_ano[q],  n_questions)
#                 if scr_ano[q] <  n_questions or self.maxlimq<len(inds_all) : break
            
# #             print('Number of questions is {}.'.format(len(inds_all)))
#             return inds_all,inds_interest

#         else:
#             assert 0,'Unknown number of questions.'

    def deactivate_feature_extractor(self):
        tlayers_index = []
        for i,lay in enumerate(self.clf.layers):
            if lay.count_params() !=0:
                tlayers_index.append(i)
        for i in tlayers_index[:-1]:
            print('#{:3d} trainable layer is {:30s} and now it is freezed!'.format(i,self.clf.layers[i].name))
            self.clf.layers[i].trainable=False

    def ask_human(self,x,y,n_questions,q_score='from_highest',predictor=None,minacc=0.0):
        yy = self.lm.to_y(y,onehot=0,add_new=1)
        norm = np.sum(list(self.interest.values()))
        inds_all = []
        inds_interest = []
        self.nqs = {}
        
        suggestions = []
        
        missed_q = False
        for intrst,w in self.interest.items():
            if intrst>=self.n_class:
                print('interest {} is out of number of class {}'.format(intrst,self.n_class))
                missed_q = True
                continue
            nqs = np.round(n_questions*w/norm).astype(int)
            nqs = max(nqs,1)
            self.nqs[intrst] = nqs
            out_obs = yy==intrst
            ano_inds = np.argwhere(out_obs)[:,0]
            
            if q_score=='from_highest':
                if predictor is None:
                    z_clf = self.clf.predict(x)
                    scr_ano = z_clf[:,intrst]
                else:
                    scr_ano = predictor(x)
                qlist = np.argsort(scr_ano)[::-1]
            elif q_score=='from_lowest':
                assert predictor is None, 'Error! You can not set the predictor while you selected q_score="from_lowest".'

                z_clf = self.clf.predict(x)
                z_clf = z_clf[:,intrst]
                z_clf[z_clf<0.5] = 0
                scr_ano = 1-2*np.abs(z_clf-0.5)
                qlist = np.argsort(scr_ano)[::-1]
                
            else:
                assert 0,'Error in the q_score definition! Available choices are "from_highest", "from_lowest".'
                
            count = 0
            for q in qlist:
                mn = 10000
                for asked in self.asked_q:
                    dist = np.sum( (x[q]-asked)**2 )
                    mn = min(mn,dist)
                if mn>minacc:
                    
#                     suggestions.append(yy[q])
                    
                    inds_all.append(q)
                    count = count+1
#                     if q in ano_inds:
#                         inds_interest.append(q)
                if count==nqs: break

#         assert n_questions==len(inds_all) or missed_q, 'problem in the numer of questions {} and {}'.format(n_questions,len(inds_all))
#         if n_questions==len(inds_all) or missed_q:
#             pass
#         else:
#             print('Warning! Problem in the numer of questions. You have chose {} but {} is asked.'.format(n_questions,len(inds_all)))
        inds_all = np.array(inds_all)
#         inds_interest = np.array(inds_interest)
#         print('==========')
#         print(inds_all,inds_interest)
#         print(suggestions)
        suggestions = yy[inds_all]
#         print(suggestions,.sum())
#         if len(inds_interest)!=0:
#             print(yy[inds_interest])
#         print('==========')
        
        
        filt_tot = np.isin(suggestions,list(self.interest.keys()))
        
        filts = []
        for interest in list(self.interest.keys()):
            filt = suggestions==interest
            filts.append(filt)
        self.tg_detaills = filts
        
        return inds_all,inds_all[filt_tot]

    
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

    def human_call(self,x,y,n_questions,q_score='from_highest',predictor=None,minacc=0.0):
#         ano_inds = np.argwhere(out2)[:,0]
#         scr_ano = self.predict(x)
#         qinds = np.argsort(scr_ano)[-n_questions:]
#         inds = np.intersect1d(qinds,ano_inds)
        inds_all,inds_interest = self.ask_human(x,y,n_questions,q_score=q_score,predictor=predictor,minacc=minacc)

        self.asked_q.extend(x[inds_all])
        if self.rmove0:
            self.x = x[inds_all]
            self.y = y[inds_all]
        else:
            self.x = np.concatenate([x[inds_all],self.x],axis=0)
            self.y = np.concatenate([y[inds_all],self.y],axis=0)
        self.n_data = self.x.shape[0]
        
        self.inds_all,self.inds_interest = inds_all,inds_interest
        true_guess = len(inds_interest)
#         print(len(inds_all),true_guess)
        return true_guess

    def predict(self,x,interest=None):
        if interest is None:
            intrst = list(self.interest.keys())[0] #self.interest
        else:
            intrst = self.lm.l2y[interest]
        z_clf = self.clf.predict(x)
        return z_clf[:,intrst]
    
    def class_predict(self,x):
        return self.clf.predict(x)
    
    def to_latent(self,x):
        z_mu = self.drt.predict(x)
        return z_mu

class LabelManager():
    def __init__(self,labels,reserved_classes=[]):
        labelsp = np.array(labels).astype(np.int).astype(str)
        assert labelsp.ndim==1,'the label array has to be 1D.' 
        labelsp = np.concatenate([reserved_classes,labelsp])
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
        for i in np.array(labels).astype(np.int).astype(str):
            if add_new:
                if i in self.l2y.keys():
                    y.append(self.l2y[i])
                else:
                    y.append(self.n_class)
            else:
                y.append(self.l2y[i])
        y = np.array(y)
        
        if onehot:
            yc = np.zeros((y.shape[0],self.n_class))
            for i in range(self.n_class):
                filt = y==i
                yc[filt,i] = 1 
            return yc
#             return to_categorical(y)
        return y

    def __call__(self,labels,onehot=False,add_new=False):
        return self.to_y(labels,onehot=onehot,add_new=add_new)
    
    def update(self,labels):
        yints = np.unique(labels).astype(np.int).astype(str)
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



# class LabelManager():
#     def __init__(self,labels):
#         labelsp = np.array(labels).astype(np.int).astype(str)
#         assert labelsp.ndim==1,'the label array has to be 1D.' 
#         self.yints = np.unique(labelsp)
#         self.l2y = {j:i for i,j in enumerate(self.yints)}
#         self.y2l = {i:j for i,j in enumerate(self.yints)}
#         self.n_class = len(self.yints)
                        
#     def to_labels(self,y):
#         yp = np.array(y)
#         if yp.ndim==2:
#             yp = np.argmax(yp,axis=1)
#         labels = []
#         for i in yp:
#             labels.append(self.y2l[i])
#         labels = np.array(labels)
#         return labels
        
#     def to_y(self,labels,onehot=False,add_new=False):
#         y = []
#         for i in np.array(labels).astype(np.int).astype(str):
#             if add_new:
#                 if i in self.l2y.keys():
#                     y.append(self.l2y[i])
#                 else:
#                     y.append(self.n_class)
#             else:
#                 y.append(self.l2y[i])
#         y = np.array(y)
        
#         if onehot:
#             return to_categorical(y)
#         return y

#     def __call__(self,labels,onehot=False,add_new=False):
#         return self.to_y(labels,onehot=onehot,add_new=add_new)
    
#     def update(self,labels):
#         yints = np.unique(labels).astype(np.int).astype(str)
#         dff = np.setdiff1d(yints,self.yints)
#         if len(dff)==0:
# #             print('No change!')
#             return False
#         l2y = {j:i+self.n_class for i,j in enumerate(dff)}
#         y2l = {i+self.n_class:j for i,j in enumerate(dff)}
        
#         self.l2y = {**self.l2y, **l2y}
#         self.y2l = {**self.y2l, **y2l}
#         self.yints = np.union1d(self.yints,dff)
#         self.n_class = len(self.yints)
#         return True


class DataContainer:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.ndata = x.shape[0]
        self.weight = []
        
    def process(self,model,c=1,wmax=None):
        y_pred = model.predict(self.x)
        delta = np.sum((self.y-y_pred)**2,axis=1)
        weight = c*(delta+1)
        weight = weight/np.min(weight)
        if not wmax is None: weight[weight>wmax] = wmax
#         weight = weight-np.min(weight)+1
        weight = weight.astype(int)
        self.weight.append(weight)
        x_w,y_w = repeat_weight_op(self.x,self.y,weight)
        return x_w,y_w    


def repeat_weight_op(x,y,weight):
    ndata = x.shape[0]
    wlx = []
    wly = []
    oped = False
    for i in np.argwhere(weight>1)[:,0]:
        w = int(weight[i])-1
        wlx.append(w*[x[i]])
        wly.append(w*[y[i]])
        oped = True
        
    if oped:
        wlx = np.concatenate(wlx,axis=0)
        wly = np.concatenate(wly,axis=0)
        x_w = np.concatenate([x,wlx],axis=0)
        y_w = np.concatenate([y,wly],axis=0)
        return x_w,y_w
    else:
        return x,y

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


