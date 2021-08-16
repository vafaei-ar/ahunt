#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pylab as plt
from os import path
import h5py as h5
from glob import glob
from tqdm import tqdm
# from skimage.io import imread
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split

from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import recall_score,precision_score,classification_report,confusion_matrix,matthews_corrcoef

import ahunt as ah
import tfcook as tfc


import argparse
parser = argparse.ArgumentParser(description='MODEL ACTIVITY ANALYZER.')
parser.add_argument('--prefix', default='res/', type=str, help='prefix')
parser.add_argument('--model', default='smp', type=str, help='model file name')
parser.add_argument('--ntry', default=5, type=int, help='try number')
parser.add_argument('--epochs', default=5, type=int, help='number of epochs')
parser.add_argument('--nqs', default=10, type=int, help='number of questions')
parser.add_argument('--nl', default=64, type=int, help='number of questions')


#     parser.add_argument('--lx', default=0, type=int, help='image length')
#     parser.add_argument('--ly', default=0, type=int, help='image width')

args = parser.parse_args()
ntry = args.ntry


# prefix = 'res/'
prefix = args.prefix
# outfile = '{}-{}'.format(prefix,postfix)

mode = args.model
n_latent = args.nl
n_questions = args.nqs # can be an integer or an array of numbers np.random.randint(3,7,n_night)
epochs = args.epochs
nl = args.nl

prefix = prefix+'nt{}-{}-nl{}'.format(ntry,mode,n_latent)

n_night = 20

noise = 0
ivc = 0
check_c = True
nightly=False
givey = True
nmin_pre=None
save_latent = False

# # Algorithm hyperparameters
num_epochs = epochs
BS = 64

aug = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.05,
    height_shift_range=0.05,
#     brightness_range=0.01,
#        shear_range=0.0,
    zoom_range=0.05,
#     horizontal_flip=True,
#     vertical_flip=True,
#     preprocessing_function=add_noise,
    fill_mode="nearest")

data = np.load('G10DE_128.npz')
x = data['x']
y_int = data['y']

x = x-x.min()
x = x/x.max()

x.shape,y_int.shape



rept = 'try: {:3d}/{:3d}  <{:10s}> | night: {:3d}/{:3d}  <{:10s}> | L2Y map: {} | Interest: {}'

def build_vgg(shape, n_class, n_latent=64):
#     tf.keras.backend.clear_session()

    baseModel = tf.keras.applications.VGG16(weights="imagenet", include_top=False,
        input_tensor=tf.keras.layers.Input(shape=shape))
    # show a summary of the base model
    print("[INFO] summary for base model...")
    #     print(baseModel.summary())

    headModel = baseModel.output
    headModel = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
    encoded = tf.keras.layers.Dense(n_latent, activation="relu")(headModel)
    xl = tf.keras.layers.Dropout(0.5)(encoded)
    output = tf.keras.layers.Dense(n_class, activation="softmax")(xl)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    encoder = keras.models.Model(inputs=baseModel.input, outputs=encoded)
    model = keras.models.Model(inputs=baseModel.input, outputs=output)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-5,
                                                                 decay_steps=10,
                                                                 decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #opt = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    #opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    loss = tf.keras.losses.CategoricalCrossentropy()

    model.compile(loss=loss, optimizer=opt,metrics=["accuracy"])
    
    return model,encoder

def get_model(shape, n_class, mode, n_latent=64):

    # tf.keras.backend.clear_session()
    if mode=='vgg':
#         baseModel = tf.keras.applications.VGG16(weights="imagenet", include_top=False,
#             input_tensor=tf.keras.layers.Input(shape=shape))
#         # show a summary of the base model
#         print("[INFO] summary for base model...")
#         #     print(baseModel.summary())

#         headModel = baseModel.output
#         headModel = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(headModel)
#         headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
#         encoded = tf.keras.layers.Dense(128, activation="relu")(headModel)
#         xl = tf.keras.layers.Dropout(0.5)(encoded)
#         output = tf.keras.layers.Dense(n_class, activation="softmax")(xl)
#         # place the head FC model on top of the base model (this will become
#         # the actual model we will train)
#         encoder = keras.models.Model(inputs=baseModel.input, outputs=encoded)
#         model = keras.models.Model(inputs=baseModel.input, outputs=output)
#         model,encoder = build_vgg(shape, n_class, n_latent=n_latent)
        
        model,encoder = build_vgg(shape, n_class, n_latent=n_latent)
        
    elif mode=='smp':
        model,encoder = ah.build_model_2dconv(shape, n_class, n_latent=n_latent, n_layers=3, kernel_size=3, pooling_size=2, l1=1e-10)
    else:
        assert 0,'Model error!'
    model.summary()
    return model,encoder 


# In[ ]:


filt = np.isin(y_int,[1,6,8]) #1,2,5,6,8,9
x = x[filt]
y_int = y_int[filt]

lm = ah.LabelManager(y_int,reserved_classes=[])

y = lm.to_y(y_int,onehot=1)

shape = x.shape[1:]
n_class = lm.n_class

#print(x.shape,y.shape)
#print(shape,n_class)
# x_train,y_train = shuffle_data(x,y)
# x_train = x_train/255
# n_test = 1500

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)

#x_test = x_test.astype(float)
#y_test = y_test.astype(int)
#x_train = x_train.astype(float)
#y_train = y_train.astype(int)
#print(x_test.shape,y_test.shape,x_train.shape,y_train.shape)


# In[ ]:


# # Dataset hyperparameters
# unlabeled_dataset_size = 100000
# labeled_dataset_size = 5000
image_size = x.shape[1]
image_channels = x.shape[3]


int_mapper = {}
lbl_mapper = {}
for i,key in enumerate(np.unique(y_int)):
    int_mapper[i] = key
    lbl_mapper[key] = i

x0 = x+0
y0 = y_int+0
x0 = x0/x0.max()


# In[ ]:


n_class,class_labels, nums = ah.describe_labels(y0,verbose=1)


# In[ ]:


# pre_data_config = {2:260,5:200,5:180,8:140,9:180}
# obs_plan = 10*[{2:120,5:90,5:80,8:60,9:80,    1:15}] + 10*[{2:120,5:90,5:80,8:60,9:80,    1:20}] #,4:10

pre_data_config = {6:500,8:400}
obs_plan = 20*[{6:66,8:50,    1:15}]# + 10*[{2:66,8:50,    1:15}] #,4:10

outlier_ind = 1
plan_tot = [pre_data_config]+obs_plan


# In[ ]:


# nmin,outlier_ind0,pre_data_config0,obs_plan0 = ah.planmaker(path,nmin_pre=nmin_pre,outlier_ind=outlier_ind)
# n_questions0 = int(0.7*nmin)

print(pre_data_config)
print(obs_plan[0])

if type(n_questions) is int or type(n_questions) is float:
    if n_questions==1 and type(n_questions) is float:
        assert 0,'Warning, when you choose 1, it should be integer.'
    n_questions = n_night*[n_questions]

if x0.ndim==3:
    n_tot,lx,ly = x0.shape
#     x = x.reshape(n_tot,lx*ly)
    x0 = x0[:,:,:,None]
    nch = 1
elif x0.ndim==4:
    n_tot,lx,ly,nch = x0.shape

if noise!=0:
    x0 += np.random.normal(0,noise,x0.shape)
n_class,class_labels, nums = ah.describe_labels(y0,verbose=0)


# In[ ]:


res_alls = []
nq_alls = []
res5s = []
res6s = []
    

for nt in range(ntry,ntry+1):
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    res5 = []
    res6 = []
    
    # if save_latent:
    z_mus = []
    lbls = []
    nq_all = []
    
    nq_all2 = []
    preds3 = []
    preds4 = []

    x,y = ah.shuffle_data(x0,y0)
    # data 0
    x, y, x_pre, y_pre = ah.data_prepare(x,y,pre_data_config)
    # observation
    obs = ah.Observetory(obs_plan,x,y)

    y = y_pre if givey else None

    ahunt = ah.AHunt(x_pre, y=y, interest=None, aug=aug)
#     ahunt.clf,ahunt.drt = ah.build_model_2dconv(ahunt.shape[1:], 
#                                                 ahunt.n_class, 
#                                                 n_latent=64, 
#                                                 n_layers=3, 
#                                                 kernel_size=3, 
#                                                 pooling_size=2, 
#                                                 l1=1e-10)
#     ahunt.clf,ahunt.drt = build_vgg(ahunt.shape[1:], ahunt.n_class, n_latent=64)
    ahunt.clf,ahunt.drt = get_model(shape=ahunt.shape[1:], n_class=ahunt.n_class, mode=mode, n_latent=n_latent)
#     build_model_2dconv(self.shape[1:],self.n_class,n_latent = 64)
    ahunt.fit(epochs=epochs,ivc=ivc)  

    ahunt0 = ah.AHunt(x_pre, y=y, interest=None, aug=aug)
#     ahunt0.clf,ahunt0.drt = ah.build_model_2dconv(ahunt0.shape[1:], 
#                                                   ahunt0.n_class, 
#                                                   n_latent=64, 
#                                                   n_layers=3, 
#                                                   kernel_size=3, 
#                                                   pooling_size=2, 
#                                                   l1=1e-10)
#     ahunt0.clf,ahunt0.drt = build_vgg(ahunt0.shape[1:], ahunt0.n_class, n_latent=64)
    ahunt0.clf,ahunt0.drt = get_model(shape=ahunt0.shape[1:], n_class=ahunt0.n_class, mode=mode, n_latent=n_latent)
    ahunt0.fit(epochs=epochs,ivc=ivc)
    
    ahunt3 = ah.AHunt(x_pre, y=y, interest=None, aug=aug)
    ahunt3.fit(epochs=epochs,ivc=ivc)

    if n_night is None: n_night = obs.n_plan
        
    phc = ah.PredictionHistoryChecker()

    for night in range(n_night):
        report = rept.format(nt,ntry,int(10*nt/ntry)*'=',
                             night,n_night,int(10*night/n_night)*'=',
                             ahunt.lm.l2y,ahunt.get_interest())
        print(report,end='\r')
        x_obs,y_obs = obs.new_obs(safemode=1,nightly=nightly)
    #     describe_labels(y_obs,verbose=1)

        out_obs = y_obs==outlier_ind
        y_true = out_obs.astype(int)
        n_anomaly = np.sum(out_obs)
        ano_inds = np.argwhere(out_obs)[:,0]

        # Method 1
        scr_ano = ah.iforest_get_anomalies(x_obs.reshape(-1,lx*ly*nch))
        trsh = np.sort(scr_ano)[-n_anomaly-1]
        y_pred = scr_ano>trsh

        rws = ah.rws_score(out_obs,scr_ano)
        rc = recall_score(y_true,y_pred)
        pr = precision_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true,y_pred)
        inds,true_guess = ah.get_tguess(n_questions[night],scr_ano,ano_inds)
        res1.append([rws,rc,pr,mcc,true_guess])

        # Method 2
        z_mu = ahunt0.to_latent(x_obs)
        scr_ano = ah.iforest_get_anomalies(z_mu)
        trsh = np.sort(scr_ano)[-n_anomaly-1]
        y_pred = scr_ano>trsh
        rws = ah.rws_score(out_obs,scr_ano)
        rc = recall_score(y_true,y_pred)
        pr = precision_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true,y_pred)
        inds,true_guess = ah.get_tguess(n_questions[night],scr_ano,ano_inds)
        res2.append([rws,rc,pr,mcc,true_guess])

        # Method 3
#         z_mu = ahunt.to_latent(x_obs)
#         if save_latent:
#             z_mus.append(z_mu)
#             lbls.append(y_obs)
#         scr_ano = ah.iforest_get_anomalies(z_mu)
#         trsh = np.sort(scr_ano)[-n_anomaly-1]
#         y_pred = scr_ano>trsh
#         rws = ah.rws_score(out_obs,scr_ano)
#         rc = recall_score(y_true,y_pred)
#         pr = precision_score(y_true,y_pred)
#         mcc = matthews_corrcoef(y_true,y_pred)

#         true_guess = phc.get_tguess(n_questions[night],scr_ano,ano_inds,x_obs)
#         res3.append([rws,rc,pr,mcc,true_guess])

        def predictor(x):
            z_mu = ahunt3.to_latent(x)
            scr_ano = ah.iforest_get_anomalies(z_mu)
            return scr_ano

        true_guess = ahunt3.human_call(x_obs,y_obs,n_questions[night],predictor=predictor)
        nq_all2.append(len(ahunt3.inds_all))

        ahunt3.fit(epochs=epochs)
        if ahunt3.n_class==4:
    #        ahunt3.set_interest({'r1':0.4,str(outlier_ind):0.6})#'1')
#             ahunt3.set_interest(str(outlier_ind))
            ahunt3.set_interest({'r1':0.5,'1':0.5})#'1')

        z_mu = ahunt3.to_latent(x_obs)
        if save_latent:
            z_mus.append(z_mu)
            lbls.append(y_obs)
        scr_ano = ah.iforest_get_anomalies(z_mu)
        trsh = np.sort(scr_ano)[-n_anomaly-1]
        y_pred = scr_ano>trsh
        preds3.append(np.c_[y_true,y_pred])
        rws = ah.rws_score(out_obs,scr_ano)
        rc = recall_score(y_true,y_pred)
        pr = precision_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true,y_pred)

        res3.append([rws,rc,pr,mcc,true_guess])

        # Method 4
        true_guess = ahunt.human_call(x_obs,y_obs,n_questions[night])
        nq_all.append(len(ahunt.inds_all))

        ahunt.fit(epochs=epochs,ivc=ivc)
        if ahunt.n_class==4:
            ahunt.set_interest({'r1':0.5,'1':0.5})#'1')
    #     model_par.append(stds_model(ahunt.clf))
        scr_ano = ahunt.predict(x_obs)
        trsh = np.sort(scr_ano)[-n_anomaly-1]
        y_pred = scr_ano>trsh
        preds4.append(np.c_[y_true,y_pred])
        rws = ah.rws_score(out_obs,scr_ano)
        rc = recall_score(y_true,y_pred)
        pr = precision_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true,y_pred)
        res4.append([rws,rc,pr,mcc,true_guess])

        if check_c and not nightly and night%5==4:

            ahunt5 = ah.AHunt(x_pre, y=y_pre, interest=None, aug=aug)
#             ahunt5.clf,ahunt5.drt = ah.build_model_2dconv(ahunt5.shape[1:], 
#                                                           ahunt5.n_class, 
#                                                           n_latent=64, 
#                                                           n_layers=3, 
#                                                           kernel_size=3, 
#                                                           pooling_size=2, 
#                                                           l1=1e-10)
#             ahunt5.clf,ahunt5.drt = build_vgg(ahunt5.shape[1:], ahunt5.n_class, n_latent=64)
            ahunt5.clf,ahunt5.drt = get_model(shape=ahunt5.shape[1:], n_class=ahunt5.n_class, mode=mode, n_latent=n_latent)
            true_guess = ahunt5.human_call(x_obs,y_obs,np.sum(nq_all))
            ahunt5.fit(epochs=night*epochs,ivc=ivc)
            if ahunt5.n_class==4:
                ahunt5.set_interest('1')

            out_obs = y_obs==outlier_ind
            y_true = out_obs.astype(int)
            n_anomaly = np.sum(out_obs)

            scr_ano = ahunt5.predict(x_obs)
            trsh = np.sort(scr_ano)[-n_anomaly-1]
            y_pred = scr_ano>trsh
            rws = ah.rws_score(out_obs,scr_ano)
            rc = recall_score(y_true,y_pred)
            pr = precision_score(y_true,y_pred)
            mcc = matthews_corrcoef(y_true,y_pred)

            res5.append([rws,rc,pr,mcc,true_guess])


            ahunt6 = ah.AHunt(x_pre, y=y_pre, interest=None, aug=aug,reserved_classes=[])
#             ahunt6.clf,ahunt6.drt = ah.build_model_2dconv(ahunt6.shape[1:], 
#                                                           ahunt6.n_class, 
#                                                           n_latent=64, 
#                                                           n_layers=3, 
#                                                           kernel_size=3, 
#                                                           pooling_size=2, 
#                                                           l1=1e-10)
#             ahunt6.clf,ahunt6.drt = build_vgg(ahunt6.shape[1:], ahunt6.n_class, n_latent=64)
            ahunt6.clf,ahunt6.drt = get_model(shape=ahunt6.shape[1:], n_class=ahunt6.n_class, mode=mode, n_latent=n_latent)
            true_guess = ahunt6.human_call(x_obs,y_obs,np.sum(nq_all))
            if true_guess==0:
                res6.append([0,0,0,0,0])
            else:

                ahunt6.fit(epochs=night*epochs,ivc=ivc)

                out_obs = y_obs==outlier_ind
                y_true = out_obs.astype(int)
                n_anomaly = np.sum(out_obs)

                scr_ano = ahunt6.predict(x_obs)
                trsh = np.sort(scr_ano)[-n_anomaly-1]
                y_pred = scr_ano>trsh
                rws = ah.rws_score(out_obs,scr_ano)
                rc = recall_score(y_true,y_pred)
                pr = precision_score(y_true,y_pred)
                mcc = matthews_corrcoef(y_true,y_pred)

                res6.append([rws,rc,pr,mcc,true_guess])
    
    res_alls.append([res1,res2,res3,res4])
    nq_alls.append(nq_all)
    res5s.append(res5)
    res6s.append(res6)
    
res_all = np.array(res_alls)
nq_all = np.array(nq_alls)
res5 = np.array(res5s)
res6 = np.array(res6s)

for i in range(4):
    res_all[:,i,:,4] = 100*res_all[:,i,:,4]/nq_all

np.savez(prefix+'GAL_res',res_all=res_all[0],
                          nq_all=nq_all[0],
                          nq_all2=nq_all2,
                          res5=res5[0],
                          res6=res6[0],
                          preds3=preds3,
                          preds4=preds4)


# In[ ]:


n_night = res_all.shape[2]
xx = np.arange(res_all.shape[2]/res5.shape[2])
xx = xx/xx.max()*n_night

cl = 95
alpha = 0.2
fig,axs = plt.subplots(2,2,figsize=(14,11))

lbls = ['iforest_raw','iforest_latent-static','iforest_latent-learning','AHunt']
clrs = ['k','r','g','b']
metric_names = ['RWS','True candidates (%)','recall','MCC']


for j,jj in enumerate([0,4,1,3]):
    ax = axs[j//2,j%2]
    for  i in range(4):
        ah.analyze_plot(ax, metric=res_all[:,i,:,jj], cl=cl, clr=clrs[i], label=lbls[i], alpha=alpha)

    ah.analyze_plot(ax, metric=res5[:,:,jj], x=xx, cl=cl, clr='orange', label='classifier1', alpha=alpha)
    ah.analyze_plot(ax, metric=res6[:,:,jj], x=xx, cl=cl, clr='yellow', label='classifier2', alpha=alpha)

    ax.set_xlabel('epoch',fontsize=15)
    ax.set_ylabel(metric_names[j],fontsize=15)
    ax.set_xlim(0,n_night-1)
    if jj==4:
        ax.set_ylim(0,102)
    else:
        ax.set_ylim(0,1.05)
    
ax.legend(fontsize=13)
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.99, top=0.99, wspace=None, hspace=None)
plt.savefig(prefix+'GAL_res.jpg',dpi=150)













