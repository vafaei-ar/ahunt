from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.keras.utils import to_categorical

def augment(aug,x):
    aug.fit(x)
    out = []
    for i in x:
        out.append(aug.random_transform(i))
    return np.array(out)

# def balance_aug(x0,y0,aug=None,mixup=False,reshape=None):
# #     assert y0.ndim>1,'Agumentor ndim problem!'
    
#     if not reshape is None:
#         shape0 = x0.shape
#         x = x0.reshape(reshape)
#     else:
#         x = x0+0
#     if y0.ndim==2:
#         y = np.argmax(y0,axis=1)
#         to_cat = True
#     elif y0.ndim==1:
#         y = y0+0
#         to_cat = False
#     else:
#         assert 0,'Agumentor y ndim problem!'
        
#     class_labels, nums = np.unique(y,return_counts=True)
#     nmax = max(nums)
#     for i,(lbl,n0) in enumerate(zip(class_labels,nums)):
#         if nmax==n0:
#             continue
#         delta = nmax-n0
#         x_sub = x[y==lbl]
#         inds = np.arange(n0)
#         nrep = (nmax//len(inds))+1
#         inds = np.repeat(inds, nrep)
#         np.random.shuffle(inds)
#         inds = inds[:delta]
#         x_sub = x_sub[inds]
#         if not aug is None:
#             x_sub = augment(aug,x_sub)
            
#         if mixup:
#             print('MIXUP is not supperted. IGNORED!')
#             pass
            
#         x = np.concatenate([x,x_sub],axis=0)
#         y = np.concatenate([y,delta*[lbl]],axis=0)
        
#     if not reshape is None:
#         x = x.reshape([-1]+list(shape0)[1:])
#     if to_cat:
#         y = to_categorical(y)
#     return x,y

def balance_aug(x0,y0,aug=None,mixup=False,reshape=None,tocater=None):
#     assert y0.ndim>1,'Agumentor ndim problem!'
    
    if not reshape is None:
        shape0 = x0.shape
        x = x0.reshape(reshape)
    else:
        x = x0+0
        
    if not tocater is None:
        y = y0+0
        to_cat = True
    elif y0.ndim==2:
        y = np.argmax(y0,axis=1)
        to_cat = True
    elif y0.ndim==1:
        y = y0+0
        to_cat = False
    else:
        assert 0,'Agumentor y ndim problem!'
        
    class_labels, nums = np.unique(y,return_counts=True)
    nmax = max(nums)
    for i,(lbl,n0) in enumerate(zip(class_labels,nums)):
        if nmax==n0:
            continue
        delta = nmax-n0
        x_sub = x[y==lbl]
        inds = np.arange(n0)
        nrep = (nmax//len(inds))+1
        inds = np.repeat(inds, nrep)
        np.random.shuffle(inds)
        inds = inds[:delta]
        x_sub = x_sub[inds]
        if not aug is None:
            x_sub = augment(aug,x_sub)
            
        if mixup:
            print('MIXUP is not supperted. IGNORED!')
            pass
            
        x = np.concatenate([x,x_sub],axis=0)
        y = np.concatenate([y,delta*[lbl]],axis=0)
        
    if not reshape is None:
        x = x.reshape([-1]+list(shape0)[1:])
    if to_cat:
        if tocater is None:
            y = to_categorical(y)
        else:
            y = tocater(y)
    return x,y

class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
    
def mixup(x0,y0,alpha,beta,num_classes=None):
    x = x0+0
    y = y0+0
    
    tocat = False
    if y.ndim==1:
        y = to_categorical(y,num_classes=num_classes)
        tocat = True
        print('The labels are converted into categorical')

    class_labels, nums = np.unique(y,return_counts=True)
    # print(class_labels, nums)

    
    nums = np.sum(y,axis=0)

    nmax = max(nums)
    # for i,(lbl,n0) in enumerate(zip(class_labels,nums)):
    for i,n0 in enumerate(nums):

        if nmax==n0 or n0==0:
            continue
        delta = int(nmax-n0)
        
        x_sub = x[y[:,i].astype(bool)]
        y_sub = y[y[:,i].astype(bool)]

        inds = np.arange(n0)
        nrep = (nmax//len(inds))
        inds = np.repeat(inds, nrep)
        np.random.shuffle(inds)
        inds = inds[:delta].astype(int)

        x_sub = x_sub[inds]
        y_sub = y_sub[inds]

        b = np.random.beta(alpha,beta,delta)[:,None]

        inds = np.arange(x.shape[0])
        np.random.shuffle(inds)
        inds = inds[:delta]
        xt = x[inds]
        yt = y[inds]

        if x.ndim==2:
            x_sub = b[:,:]*x_sub+(1-b[:,:])*xt
        elif x.ndim==3:
            x_sub = b[:,:,None]*x_sub+(1-b[:,:,None])*xt
        elif x.ndim==4:
            x_sub = b[:,:,None,None]*x_sub+(1-b[:,:,None,None])*xt
        else:
            assert 0,'The shape is not as expected! {}-{}'.format(x.shape,x_sub.shape)
        
        y_sub = b*y_sub+(1-b)*yt

        x = np.concatenate([x,x_sub],axis=0)
        y = np.concatenate([y,y_sub],axis=0)
#     if tocat:
#         y = np.argmax(y,axis=1)
    return x,y


class DataFeed:
    def __init__(self,x,y,aug = None):
        self.x = x
        self.y = ySelmakhodadadiSelmakhodadadi
        self.aug = aug
        self.nd,self.nx,self.ny,self.ch = x.shape
        self.banance()
        
    def banance(self):
        self.xb,self.yb = balance_aug(self.x,self.y,aug=self.aug)
        self.ndb = self.xb.shape[0]
    def __call__(self,num,reset=False):
        if reset:
            self.banance()
        inds = np.arange(self.ndb)
        np.random.shuffle(inds)
        inds = inds[:num]
        return self.xb[inds],self.yb[inds]

