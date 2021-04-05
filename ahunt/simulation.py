from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pylab as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,precision_score,classification_report,confusion_matrix,matthews_corrcoef
from sklearn.ensemble import IsolationForest

from .data_utils import load_npz,data_prepare,describe_labels,rws_score,analyze,check_int,check_float
from .models import stds_model
from .ahunt_man import AHunt

BOLD_BEGIN = '\033[1m'
BOLD_END   = '\033[0m' 

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


def compare(path,
            noise,
            outlier_ind,
            pre_data_config,
            obs_plan,
            n_questions,
            aug = None,
            n_night = None,
            nightly = False,
            epochs = 5,
            givey = True
           ):
    
    if type(n_questions) is int or type(n_questions) is float:
        if n_questions==1 and type(n_questions) is float:
            assert 0,'Warning, when you choose 1, it should be integer.'
        n_questions = n_night*[n_questions]
    print(BOLD_BEGIN+'Loaded data configuration:'+BOLD_END)
    x,y,int_mapper,lbl_mapper = load_npz(path,verbose=0)
    x = x/x.max()
    
    if x.ndim==3:
        n_tot,lx,ly = x.shape
    #     x = x.reshape(n_tot,lx*ly)
        x = x[:,:,:,None]
        nch = 1
    elif x.ndim==4:
        n_tot,lx,ly,nch = x.shape
        
    if noise!=0:
        x += np.random.normal(0,noise,x.shape)
    n_class,class_labels, nums = describe_labels(y,verbose=0)
#     print(n_class)
    print(BOLD_BEGIN+' ******** previous dataset ******** '+BOLD_END)

    x, y, x_pre, y_pre = data_prepare(x,y,pre_data_config)
    describe_labels(y_pre,verbose=0)
    # x_obs = x_obs_orig.reshape(x_obs_orig.shape[0],-1)
    # y_obs1 = y_obs_orig+0

    print(BOLD_BEGIN+' ******** Observation begins ******** '+BOLD_END)
    obs = Observetory(obs_plan,x,y)

    rws1,rws2,rws3,rws4 = [],[],[],[]
    tg1,tg2,tg3,tg4 = [],[],[],[]
    rc1,rc2,rc3,rc4 = [],[],[],[]
    pr1,pr2,pr3,pr4 = [],[],[],[]
    mcc1,mcc2,mcc3,mcc4 = [],[],[],[]
#     print(n_class)
#     clf,drt = build_model(shape=(lx*ly),n_class=n_class,n_latent = 64)
#     ahunt = AHunt(x_pre, y_pre,clf,drt,interest=outlier_ind,aug=aug)
    if givey:
        ahunt = AHunt(x_pre, y_pre,interest=None,aug=aug)
    else:
        ahunt = AHunt(x_pre, y=None,interest=None,aug=aug)
    ahunt.fit()

#     clf0,drt0 = build_model(shape=(lx*ly),n_class=n_class,n_latent = 64)
    if givey:
        ahunt0 = AHunt(x_pre, y_pre,interest=None,aug=aug)
    else:
        ahunt0 = AHunt(x_pre, y=None,interest=None,aug=aug)
    ahunt0.fit()

    model_par = []
    qinds3,seen3 = [],[]
    z_mus = []
    lbls = []
    model_par.append(stds_model(ahunt.clf))
    
    if n_night is None: n_night = obs.n_plan
    
    for night in range(n_night):
        x_obs,y_obs = obs.new_obs(safemode=1,nightly=nightly)
    #     describe_labels(y_obs,verbose=1)

        out_obs = y_obs==outlier_ind
        y_true = out_obs.astype(int)
        n_anomaly = np.sum(out_obs)
        ano_inds = np.argwhere(out_obs)[:,0]

        # Method 1
#         print(y_obs.shape)
#         print(out_obs.shape)
#         print(x_obs.shape)
        
        scr_ano = iforest_get_anomalies(x_obs.reshape(-1,lx*ly*nch))
        trsh = np.sort(scr_ano)[-n_anomaly-1]
        y_pred = scr_ano>trsh
        
        
#         print(out_obs.shape)
#         print(scr_ano.shape)
        
        
        rws = rws_score(out_obs,scr_ano)
        rc = recall_score(y_true,y_pred)
        pr = precision_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true,y_pred)
        rws1.append(rws)
        rc1.append(rc)
        pr1.append(pr)
        mcc1.append(mcc)
#         assert 0

        if check_int(n_questions[night]):
            qinds = np.argsort(scr_ano)[-n_questions[night]:]
        elif check_float(n_questions[night]) and n_questions[night]<1:
#             np.where(x<5)[0]
            qinds = np.where(scr_ano > n_questions[night])[0]
#             qinds = np.argsort(scr_ano)#[-:]
        else:
            print(type(n_questions[night]))
            print(n_questions[night])
            assert 0,'Unknown number of questions.'
    
        inds = np.intersect1d(qinds,ano_inds)
        true_guess = len(inds)
        tg1.append(true_guess)
        
        # Method 2
        
        z_mu = ahunt0.to_latent(x_obs)
        scr_ano = iforest_get_anomalies(z_mu)
        trsh = np.sort(scr_ano)[-n_anomaly-1]
        y_pred = scr_ano>trsh
        rws = rws_score(out_obs,scr_ano)
        rc = recall_score(y_true,y_pred)
        pr = precision_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true,y_pred)
        rws2.append(rws)
        rc2.append(rc)
        pr2.append(pr)
        mcc2.append(mcc)
        
#         qinds = np.argsort(scr_ano)[-n_questions[night]:]
        if check_int(n_questions[night]):
            qinds = np.argsort(scr_ano)[-n_questions[night]:]
        elif check_float(n_questions[night]) and n_questions[night]<1:
#             np.where(x<5)[0]
            qinds = np.where(scr_ano > n_questions[night])[0]
#             qinds = np.argsort(scr_ano)#[-:]
        else:
            print(n_questions[night])
            assert 0,'Unknown number of questions.'
        
        inds = np.intersect1d(qinds,ano_inds)
        true_guess = len(inds)
        tg2.append(true_guess)

        # Method 3
        
        z_mu = ahunt.to_latent(x_obs)
        z_mus.append(z_mu)
        lbls.append(y_obs)
        scr_ano = iforest_get_anomalies(z_mu)
        trsh = np.sort(scr_ano)[-n_anomaly-1]
        y_pred = scr_ano>trsh
        rws = rws_score(out_obs,scr_ano)
        rc = recall_score(y_true,y_pred)
        pr = precision_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true,y_pred)
        rws3.append(rws)
        rc3.append(rc)
        pr3.append(pr)
        mcc3.append(mcc)

        cands = np.argsort(scr_ano)[::-1]
        ncand = 0
        true_guess = 0
        for ic in cands:
#             dist3 = []
#             for dd in seen3:
#                 dist3.append(np.sum( (x_obs[ic]-dd)**2 ))
            mn = 10000
            for asked in seen3:
                dist = np.sum( (x_obs[ic]-asked)**2 )
                mn = min(mn,dist)
            if mn>1e-3:
                if ic in ano_inds:
                    true_guess = true_guess+1
#                 qinds3.append(ic) 
                seen3.append(x_obs[ic])
                ncand = ncand+1
        
        
            if check_int(n_questions[night]):
                if ncand==n_questions[night]: break
#                     break
#                 if len(inds_all)==n_questions: break
            elif check_float(n_questions[night]) and n_questions[night]<1:
                if scr_ano[ic] <  n_questions[night]: break
            else:
                print(n_questions[night])
                assert 0,'Unknown number of questions.'

#         taken3.extend(qinds)
#         inds = np.intersect1d(qinds,ano_inds)
#          = len(inds)
        tg3.append(true_guess)
        
        # Method 4
#         true_guess = ano_hunt.human_call1(x_obs,y_obs,n_questions[night])
        true_guess = ahunt.human_call(x_obs,y_obs,n_questions[night])
        tg4.append(true_guess)

        ahunt.fit()
        model_par.append(stds_model(ahunt.clf))
        scr_ano = ahunt.predict(x_obs)
        trsh = np.sort(scr_ano)[-n_anomaly-1]
        y_pred = scr_ano>trsh
        rws = rws_score(out_obs,scr_ano)
        rc = recall_score(y_true,y_pred)
        pr = precision_score(y_true,y_pred)
        mcc = matthews_corrcoef(y_true,y_pred)
        rws4.append(rws)
        rc4.append(rc)
        pr4.append(pr)
        mcc4.append(mcc)
        
#         print('this',y_true.sum(),y_pred.sum())
#         print(confusion_matrix(y_true, y_pred))
#         print(classification_report(y_true,y_pred))
#     assert 0
    rwss = [rws1,rws2,rws3,rws4]
    tgs = [tg1,tg2,tg3,tg4]
    rcs = [rc1,rc2,rc3,rc4]
    prs = [pr1,pr2,pr3,pr4]
    mccs = [mcc1,mcc2,mcc3,mcc4]
    
    return rwss,tgs,rcs,prs,mccs,z_mus,lbls,model_par

def planmaker(path,nmin_pre=None,outlier_ind=None):

    data = np.load(path)
    shapes = []
    keys = []
    for key in list(data):
        shapes.append(data[key].shape)
        keys.append(key)
    shapes = np.array(shapes)
    inds_sorted = np.argsort(shapes[:,0])
    outlier_indp = inds_sorted[0]
    
    if outlier_ind is None or outlier_indp==outlier_ind:
        outlier_ind = outlier_indp
        nmin = shapes[inds_sorted[0]][0]//25
        nmaj = shapes[inds_sorted[1]][0]//25
        nmin = min(nmin,nmaj//10)
    else:
        nmin = shapes[inds_sorted[0]][0]//25
        nmaj = shapes[inds_sorted[1]][0]//25
        nmin = min(nmin,nmaj//10)    
    print('outlier is ',keys[outlier_ind])
    if nmin_pre is None:
        nmin_pre = 3*nmin
    pre_data_config = {i:(i==outlier_ind)*nmin_pre+3*(i!=outlier_ind)*nmaj for i in range(len(list(data)))}
    obs_plan = 20*[{i:(i==outlier_ind)*nmin+(i!=outlier_ind)*nmaj for i in range(len(list(data)))}]
    return nmin,outlier_ind,pre_data_config,obs_plan

def run_for(fname,
            epochs = 5,
            noise = 0.0,
            n_night = 30,
            nightly=False,
            ntry = 5,
            givey = True,
            n_questions = None,
            outlier_ind = None,
            pre_data_config = None,
            obs_plan = None,
            aug = None,
            nmin_pre=None,
            prefix = ''):
    # n_questions = 7
    # pre_data_config = {0:100,1:1}
    # obs_plan = 10*[{0:100,1:10}]
    # outlier_ind = 1
    path = '/home/vafaeisa/scratch/datasets/prepared/{}.npz'.format(fname)

    nmin,outlier_ind0,pre_data_config0,obs_plan0 = planmaker(path,nmin_pre=nmin_pre,outlier_ind=outlier_ind)
    n_questions0 = int(0.7*nmin)
    
    if n_questions is None: n_questions=n_questions0
    if outlier_ind is None: outlier_ind=outlier_ind0
    if pre_data_config is None: pre_data_config=pre_data_config0
    if obs_plan is None: obs_plan=obs_plan0
    print(pre_data_config)
    print(obs_plan[0])

    rws1s,rws2s,rws3s,rws4s = [],[],[],[]
    tg1s,tg2s,tg3s,tg4s  = [],[],[],[]
    rc1s,rc2s,rc3s,rc4s = [],[],[],[]
    pr1s,pr2s,pr3s,pr4s = [],[],[],[]
    mcc1s,mcc2s,mcc3s,mcc4s = [],[],[],[]

    z_mus,lbls,model_pars = [],[],[]
    for _ in range(ntry):
        res = compare(path = path,
                      noise = noise,
                      outlier_ind = outlier_ind,
                      pre_data_config = pre_data_config,
                      obs_plan = obs_plan,
                      n_questions = n_questions,
                      aug = aug,
                      n_night=n_night,
                      nightly=nightly,
                      epochs = epochs,
                      givey = givey 
                     )

        rwss,tgs,rcs,prs,mccs,z_mu,lbl,model_par = res

        rws1,rws2,rws3,rws4 = rwss
        tg1,tg2,tg3,tg4 = tgs
        rc1,rc2,rc3,rc4 = rcs
        pr1,pr2,pr3,pr4 = prs
        mcc1,mcc2,mcc3,mcc4 = mccs

        rws1s.append(rws1)
        rws2s.append(rws2)
        rws3s.append(rws3)
        rws4s.append(rws4)
        tg1s.append(tg1)
        tg2s.append(tg2)
        tg3s.append(tg3)
        tg4s.append(tg4)
        rc1s.append(rc1)
        rc2s.append(rc2)
        rc3s.append(rc3)
        rc4s.append(rc4)
        pr1s.append(pr1)
        pr2s.append(pr2)
        pr3s.append(pr3)
        pr4s.append(pr4)
        mcc1s.append(mcc1)
        mcc2s.append(mcc2)
        mcc3s.append(mcc3)
        mcc4s.append(mcc4)
        z_mus.append(z_mu)
        lbls.append(lbl)
        model_pars.append(model_par)

    np.savez('{}{}_res'.format(prefix,fname),
            rws1s = np.array(rws1s),
            rws2s = np.array(rws2s),
            rws3s = np.array(rws3s),
            rws4s = np.array(rws4s),

            tg1s = np.array(tg1s),
            tg2s = np.array(tg2s),
            tg3s = np.array(tg3s),
            tg4s = np.array(tg4s),

            rc1s = np.array(rc1s),
            rc2s = np.array(rc2s),
            rc3s = np.array(rc3s),
            rc4s = np.array(rc4s),

            pr1s = np.array(pr1s),
            pr2s = np.array(pr2s),
            pr3s = np.array(pr3s),
            pr4s = np.array(pr4s),

            mcc1s = np.array(mcc1s),
            mcc2s = np.array(mcc2s),
            mcc3s = np.array(mcc3s),
            mcc4s = np.array(mcc4s),
            z_mus = np.array(z_mus),
            lbls = np.array(lbls),
            model_pars = np.array(model_pars)
           )

    return n_questions
    
def plot_for(fname,n_questions,prefix=''):
    alpha = 0.2
    data = np.load('{}{}_res.npz'.format(prefix,fname),allow_pickle=1)
#     for i in list(data):
#         print(i)
#         exec("{}=np.array(data['{}'])".format(i,i),locals=locals)

    rws1s = data['rws1s']
    rws2s = data['rws2s']
    rws3s = data['rws3s']
    rws4s = data['rws4s']

    tg1s =  data['tg1s']
    tg2s =  data['tg2s']
    tg3s =  data['tg3s']
    tg4s =  data['tg4s']

    rc1s =  data['rc1s']
    rc2s =  data['rc2s']
    rc3s =  data['rc3s']
    rc4s =  data['rc4s']

    pr1s =  data['pr1s']
    pr2s =  data['pr2s']
    pr3s =  data['pr3s']
    pr4s =  data['pr4s']

    mcc1s = data['mcc1s']
    mcc2s = data['mcc2s']
    mcc3s = data['mcc3s']
    mcc4s = data['mcc4s']
    z_mus = data['z_mus']
    lbls = data['lbls']
    model_pars = data['model_pars']
        
             
#     fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))

#     clrs = ['r','b','g']
#     for i in range(3):
#         clr = clrs[i]

#         m,l,u = analyze(model_pars[:,:,0,i])
#         ax1.plot(m,clr,label='layer='+str(i))
#         ax1.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

#         m,l,u = analyze(model_pars[:,:,1,i])
#         ax2.plot(m,clr,label='layer='+str(i))
#         ax2.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

#     ax1.set_xlabel('night',fontsize=15)
#     ax2.set_xlabel('night',fontsize=15)
#     ax1.set_ylabel('mean weights',fontsize=15)
#     ax2.set_ylabel('mean bias',fontsize=15)
#     ax1.legend(fontsize=13)

#     ax1.set_xlim(0,m.shape[0]-1)
#     # ax1.set_ylim(0,1)
#     ax2.set_xlim(0,m.shape[0]-1)
#     # ax2.set_ylim(0,5.5)

#     plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.99, wspace=None, hspace=None)
#     plt.savefig('{}layers_{}.jpg'.format(prefix,fname),dpi=150)
#     plt.close()

             
    fig,axs = plt.subplots(2,2,figsize=(14,10))

    ax = axs[0,0]

    m,l,u = analyze(rws1s)
    clr = 'k'
    ax.plot(m,clr,label='iforest_raw')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(rws2s)
    clr = 'r'
    ax.plot(m,clr,label='iforest_latent-static')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(rws3s)
    clr = 'g'
    ax.plot(m,clr,label='iforest_latent-learning')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(rws4s)
    clr = 'b'
    ax.plot(m,clr,label='AHunt')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    ax.set_xlabel('night',fontsize=15)
    ax.set_ylabel('RWS',fontsize=15)

    ax.legend(fontsize=13)
    ax.set_xlim(0,m.shape[0]-1)
    ax.set_ylim(0,1)

    ax = axs[0,1]
    
    m,l,u = analyze(100*tg1s/n_questions)
    clr = 'k'
    ax.plot(m,clr)
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(100*tg2s/n_questions)
    clr = 'r'
    ax.plot(m,clr)
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(100*tg3s/n_questions)
    clr = 'g'
    ax.plot(m,clr)
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)
    
    m,l,u = analyze(100*tg4s/n_questions)
    clr = 'b'
    ax.plot(m,clr)
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    ax.set_xlabel('night',fontsize=15)
    ax.set_ylabel('True candidates (%)',fontsize=15)

    ax.set_xlim(0,m.shape[0]-1)
    ax.set_ylim(0,102)


    ax = axs[1,0]

    m,l,u = analyze(rc1s)
    clr = 'k'
    ax.plot(m,clr,label='iforest_raw')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(rc2s)
    clr = 'r'
    ax.plot(m,clr,label='iforest_latent-static')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(rc3s)
    clr = 'g'
    ax.plot(m,clr,label='iforest_latent-learning')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(rc4s)
    clr = 'b'
    ax.plot(m,clr,label='AHunt')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    ax.set_xlabel('night',fontsize=15)
    ax.set_ylabel('recall',fontsize=15)

#     ax.legend(fontsize=13)
    ax.set_xlim(0,m.shape[0]-1)
    ax.set_ylim(0,1)


    ax = axs[1,1]

    # m,l,u = analyze(pr1s)
    # clr = 'y'
    # ax.plot(m,clr,label='iforest_raw')
    # ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    # m,l,u = analyze(pr2s)
    # clr = 'r'
    # ax.plot(m,clr,label='iforest_latent-static')
    # ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    # m,l,u = analyze(pr3s)
    # clr = 'g'
    # ax.plot(m,clr,label='iforest_latent-learning')
    # ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    # m,l,u = analyze(pr4s)
    # clr = 'b'
    # ax.plot(m,clr,label='AnoHunt')
    # ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    # ax.set_xlabel('night',fontsize=15)
    # ax.set_ylabel('precision',fontsize=15)

    m,l,u = analyze(mcc1s)
    clr = 'k'
    ax.plot(m,clr,label='iforest_raw')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(mcc2s)
    clr = 'r'
    ax.plot(m,clr,label='iforest_latent-static')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(mcc3s)
    clr = 'g'
    ax.plot(m,clr,label='iforest_latent-learning')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    m,l,u = analyze(mcc4s)
    clr = 'b'
    ax.plot(m,clr,label='AHunt')
    ax.fill_between(np.arange(m.shape[0]),l,u,color=clr,alpha=alpha)

    ax.set_xlabel('night',fontsize=15)
    ax.set_ylabel('MCC',fontsize=15)

#     ax.legend(fontsize=13)
    ax.set_xlim(0,m.shape[0]-1)
    ax.set_ylim(0,1)


    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.99, wspace=None, hspace=None)
    plt.savefig('{}result_{}.jpg'.format(prefix,fname),dpi=150)
    
    return z_mus,lbls

def iforest_get_anomalies(z):
    isof = IsolationForest()
    isof.fit(z)
    scores_pred = isof.decision_function(z)
    scores_pred = scores_pred.max()-scores_pred
    return scores_pred
