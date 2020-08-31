import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use('Agg')
import numpy as np
import math
import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from air_model_dis_deep_2 import DrawModel
from config_multi_dsprites import *
from utility import Variable,save_image, save_image_single,xrecons_grid
import torch.nn.utils
import matplotlib.pyplot as plt
from functools import partial
import visdom
import itertools
from load_data import load_data_label 
from observations import multi_mnist
from viz import draw_many, tensor_to_objs, latents_to_tensor

cuda = True
from_scratch = True
model_save_path = 'save/weights_210000.tar'
data_path = '/local/scratch/wd263/my_dsprites/'

torch.set_default_tensor_type('torch.FloatTensor')

device = torch.device("cuda" if cuda else "cpu")

#train_loader, test_loader = load_data(data_path,batch_size)
def count_accuracy(X, true_counts, air, batch_size):
    assert X.size(0) == true_counts.size(0), 'Size mismatch.'
    assert X.size(0) % batch_size == 0, 'Input size must be multiple of batch_size.'
    counts = torch.LongTensor(4, 5).zero_()
    error_latents = []
    error_indicators = []
    #print(type(X))

    def count_vec_to_mat(vec, max_index):
        out = torch.LongTensor(vec.size(0), max_index + 1).zero_()
        out.scatter_(1, vec.type(torch.LongTensor).view(vec.size(0), 1), 1)
        return out

    for i in range(X.size(0) // batch_size):
        #print(i * batch_size,(i + 1) * batch_size)
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        X_batch = X_batch.to(device)
        true_counts_batch = true_counts[i * batch_size:(i + 1) * batch_size]
        z_where, z_pres,z_cat = air.forward(X_batch, 0.5)

        z_cat_numpy = torch.stack(z_cat).detach().cpu().permute((1,0,2)).contiguous().numpy()

        #some test
        z_cat_t = torch.stack(z_cat).detach().cpu().permute((1,0,2))
        z_cat_tc = torch.stack(z_cat).detach().cpu().permute((1,0,2)).contiguous()
        #print((z_cat_t-z_cat_tc).data)
        z_cat_z = z_cat[0].detach().cpu().numpy()
        #print(np.std(z_cat_z - z_cat_numpy[:,0,:]))
        #print(torch.stack(z_pres).size())
        z_pres_numpy = torch.round(torch.stack(z_pres)).detach().cpu().permute((1,0,2)).contiguous().numpy()
        z_pres_numpy = np.squeeze(z_pres_numpy)
        if i==0:
            z_cat_array = z_cat_numpy
            z_pres_array = z_pres_numpy
        #else if i<10:
        #    z_cat_array = np.concatenate([z_cat_array,z_cat_numpy],0)
        #    z_pres_array = np.concatenate([z_pres_array,z_pres_numpy],0)
         
        inferred_counts = sum(torch.round(z).cpu() for z in z_pres).squeeze().data
        true_counts_m = count_vec_to_mat(true_counts_batch, 3)
        inferred_counts_m = count_vec_to_mat(inferred_counts, 4)
        counts += torch.mm(true_counts_m.t(), inferred_counts_m)
        error_ind = 1 - (true_counts_batch == inferred_counts)
        error_ix = error_ind.nonzero().squeeze()
        error_latents.append(latents_to_tensor((z_where, z_pres)).index_select(0, error_ix))
        error_indicators.append(error_ind)

    acc = counts.diag().sum().float() / X.size(0)
    error_indices = torch.cat(error_indicators).nonzero().squeeze()
    if X.is_cuda:
        error_indices = error_indices.cuda()
    return acc, counts, torch.cat(error_latents), error_indices, z_cat_array, z_pres_array

def searchsorted2d(a,b):
    m,n = a.shape
    max_num = np.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num*np.arange(a.shape[0])[:,None]
    p = np.searchsorted( (a+r).ravel(), (b+r).ravel() ).reshape(m,-1)
    return p - n*(np.arange(m)[:,None])

def numpy_isin2D(A,B):
    sB = np.sort(B,axis=1)
    idx = searchsorted2d(sB,A)
    idx[idx==sB.shape[1]] = 0
    return np.take_along_axis(sB, idx, axis=1) == A

def permute_c(c,perm):
    #print(type(c),c.dtype)
    permuted_c = np.zeros_like(c)
    permuted_c = perm[c.astype(np.int)]
    #for i in range(c.shape[0]):
    #    permuted_c[i]=perm[c[i]]
    return permuted_c

def compute_correspondence(discrete_z, z_pres, c, c_mask):
    print(discrete_z.shape,c.shape)
    z_idx = np.argmax(discrete_z,axis=-1)
    num_pred = np.sum(z_pres)
    print(num_pred)
    c = c.astype(np.int)
    z_idx_masked = np.multiply(z_idx,z_pres) + z_pres - 1
    #c_masked = np.multiply(c,z_pres)
    #print(z_pres)
    #print(z_idx_masked)
    #print(c_masked)    
    best_score = 0
    best_perm = np.zeros(3)
    for perm in itertools.permutations([0,1,2]):
        #print(perm)
        permuted_c = permute_c(c,np.array(perm))
        c_masked = np.multiply(permuted_c,z_pres) + 2 * (c_mask + z_pres -1)

        #print('z_idx',z_idx_masked[:10])
        #print('c',c_masked[:10])
        #score_array = np.where( np.logical_and(np.mean(z_idx_masked,axis=-1) == np.mean(c_masked,axis=-1),np.var(z_idx_masked,axis=-1) == np.var(c_masked,axis=-1)),1,0)
        #score = np.sum(score_array)
        isin_array = numpy_isin2D(z_idx_masked,c_masked)
        #print('isin',isin_array)
        score = np.sum(isin_array)
        #for i in range(c.shape[0]):
        #    score += np.sum(np.isin(z_idx_masked[i],c[i]))
        #print(best_score,score)
        
        #print(np.mean(z_idx_masked,axis=-1), np.mean(c_masked,axis=-1), np.var(z_idx_masked,axis=-1), np.var(c_masked,axis=-1))
        #score = np.sum(np.multiply(np.clip(np.abs(z_idx-permuted_c),0,1),z_pres))
        if score > best_score:
            best_score = score
            best_perm = perm


    print('corresponding classes:',best_perm)
    print('coorespondence rate= ',best_score/num_pred,best_score,num_pred) 
    return best_score/num_pred


def make_prior(k):
    assert 0 < k <= 1
    u = 1 / (1 + k + k**2 + k**3)
    p0 = 1 - u
    p1 = 1 - (k * u) / p0
    p2 = 1 - (k**2 * u) / (p0 * p1)
    p3 = 1 - (k**3 * u) / (p0 * p1 * p2)
    trial_probs = [p0, p1, p2, p3]
    print('trial:',trial_probs)
    # dist = [1 - p0, p0 * (1 - p1), p0 * p1 * (1 - p2), p0 * p1 * p2]
    # print(dist)
    return lambda t: trial_probs[t]

def lin_decay(initial, final, begin, duration, t):
    assert duration > 0
    x = (final - initial) * (t - begin) / duration + initial
    return max(min(x, initial), final)


def exp_decay(initial, final, begin, duration, t):
    assert final > 0
    assert duration > 0
    # half_life = math.log(2) / math.log(initial / final) * duration
    decay_rate = math.log(initial / final) / duration
    x = initial * math.exp(-decay_rate * (t - begin))
    return max(min(x, initial), final)


train_loader, X, true_counts, labels, labels_mask = load_data_label(data_path,batch_size,'dsprites_data_l.npz')
print('open Visdom')
vis = visdom.Visdom(env='AIR_dspritesMY')

print('construct z_pres_prior')
base_z_pres_prior_p = make_prior(z_pres_prior)

    # Wrap with logic to apply any annealing.
def z_pres_prior_p(opt_step, time_step):
    p = base_z_pres_prior_p(time_step)
    if anneal_prior == 'None':
       return math.log(p)
    else:
       decay = dict(lin=lin_decay, exp=exp_decay)[anneal_prior]
       prior = decay(p, anneal_prior_to, anneal_prior_begin,anneal_prior_duration, opt_step)
       return math.log(prior)

model = DrawModel(T,A,B,z_size,cat_size,window_size,rnn_size)
if not from_scratch:
    model.load_state_dict(torch.load(model_save_path))
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

if USE_CUDA:
    model.cuda()

def train():
    avg_loss = 0
    count = 0
    epoch_loglik = 0

    count_acc_history = []
    corr_rate_history = []
    for i in range(6): 
        print(X[i*64:(i+1)*64].shape)
        save_image_single(X[i*64:(i+1)*64],i,'X',path='X_image/')
    
    for epoch in range(epoch_num):
        for batch_idx, (data,_) in enumerate(train_loader):
            bs = data.size()[0]
            data = data.to(device)
            #data = Variable(data).view(bs, -1)
            optimizer.zero_grad()
            tau=max(tau_final, np.exp(-anneal_rate*epoch))
            loss, loglik = model.loss(data,tau, z_pres_prior_p=partial(z_pres_prior_p, count))
            epoch_loglik += loglik.cpu().data.numpy()
            avg_loss += loss.cpu().data.numpy()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            count += 1
            if count % 100 == 0:
                print('Epoch-{}; Count-{}; loss: {} tau : {};'.format(epoch, count, avg_loss / 100, tau))


                if count % 1000 == 0:
                    model.eval()
                    generate_image(count, tau)
                    acc, counts, error_z, error_ix,z_cat_array, z_pres_array = count_accuracy(X, true_counts, model, 500)
                    count_acc_history.append(acc)
                    np.savetxt('count_acc_history_cat_d.csv',np.array(count_acc_history),delimiter=',')
                    print('i={}, accuracy={}, counts={}'.format(count, acc, counts.numpy().tolist()))
                    
                    if count % 10000 == 0:
                        #acc, counts, error_z, error_ix,z_cat_array, z_pres_array = count_accuracy(X, true_counts, model, 500)
                        corr_rate = compute_correspondence(z_cat_array, z_pres_array,labels[:500],labels_mask[:500])
                        corr_rate_history.append(corr_rate)
                        np.savez_compressed('acc_history/acc_history_{}'.format(count),z_cat=z_cat_array,z_pres=z_pres_array,labels=labels[:500],labels_mask=labels_mask[:500])
                        #print('i={}, accuracy={}, counts={}'.format(count, acc, counts.numpy().tolist()))
                        
                        if count>30000:
                            torch.save(model.state_dict(),'save/weights_2.tar')
                    model.train()

                avg_loss = 0
        print('epoch {} loglik: {}'.format(epoch,epoch_loglik/count))

  
    #torch.save(model.state_dict(), 'save/weights_final.tar')
    #model.eval()
    #generate_image(count)
    np.savetxt('count_acc_history_cat_d.csv',np.array(count_acc_history),delimiter=',')
    np.savetxt('corr_rate_history_cat_d.csv',np.array(corr_rate_history),delimiter=',')

def generate_image(count,tau, gen_batch_size=64):
    #x = model.generate(batch_size)
    train_iter = iter(train_loader)
    (data,_) = train_iter.next()
    data = data.to(device)
    recon_x,z_pres,cats = model.recon(data, tau)
    z_zip = model.get_where_pres()
    z_obj = tensor_to_objs(latents_to_tensor(z_zip)[:5])
    #print(len(recon_x))
    recon_x_final = recon_x[-1]
    print(np.min(recon_x_final),np.max(recon_x_final))
    cats_array = []
    z_pres_array = []
    
    if count >= 60000:
        for i in range(3):
            cats_numpy = cats[i].detach().cpu().numpy()
            index = np.argmax(cats_numpy,axis=-1)

            z_pres_np = np.squeeze(torch.round(z_pres[i].detach()).cpu().numpy())
            z_pres_np_raw = np.squeeze(z_pres[i].detach().cpu().numpy())
            cats_array.append(cats_numpy)
            z_pres_array.append(z_pres_np_raw)

            print(index.shape,z_pres_np.shape)
            index = index * z_pres_np + z_pres_np - 1
            print('i=',i,' cat index:')
            print(np.reshape(index,(8,8)))

    np.savez_compressed('codes/recon_code_{}.csv'.format(count),cats = np.array(cats_array), z_pres = np.array(z_pres_array))


    vis.images(draw_many(data[:5].view(-1,A,B), z_obj))
            # Show reconstructions of data.
    vis.images(draw_many(torch.tensor(recon_x_final[:5]).view(-1,A,B), z_obj))
    #save_image(x,count,'gen')
    save_image(recon_x,count,'recon',path='image/')
    save_image_single(data.cpu().numpy(),count,'origin',path='image/')
    #####generate image from scratch
    #first_obj = np.random.randint(cat_size)
    first_obj = 0
    first_obj_tensor = torch.zeros(gen_batch_size,cat_size).scatter_(1,torch.tensor(first_obj).expand(gen_batch_size,1),1).float().to(device)

    #second_obj = np.random.randint(cat_size)
    second_obj = 1
    second_obj_tensor = torch.zeros(gen_batch_size,cat_size).scatter_(1,torch.tensor(second_obj).expand(gen_batch_size,1),1).float().to(device)        

    #third_obj = np.random.randint(cat_size)
    third_obj = 2
    third_obj_tensor = torch.zeros(gen_batch_size,cat_size).scatter_(1,torch.tensor(third_obj).expand(gen_batch_size,1),1).float().to(device)     

    first_scale = np.random.uniform(1.5,4,size=(gen_batch_size,1))
    first_pos = np.random.uniform(-0.3,0.3,size=(gen_batch_size,2))
    second_scale = np.random.uniform(1.5,4,size=(gen_batch_size,1))
    second_pos = np.random.uniform(-0.3,0.3,size=(gen_batch_size,2))    
    third_scale = np.random.uniform(1.5,4,size=(gen_batch_size,1))
    third_pos = np.random.uniform(-0.3,0.3,size=(gen_batch_size,2)) 
    #third_pos = np.concatenate([np.random.uniform(0.6,0.8,size=(gen_batch_size,1)),np.random.uniform(-0.6,-0.8,size=(gen_batch_size,1))],axis=-1)
    first_z_where = torch.from_numpy(np.concatenate([first_scale,first_pos],axis=-1)).float().to(device)
    second_z_where = torch.from_numpy(np.concatenate([second_scale,second_pos],axis=-1)).float().to(device)
    third_z_where = torch.from_numpy(np.concatenate([third_scale,third_pos],axis=-1)).float().to(device)

    gen_x = model.generate([first_obj_tensor,second_obj_tensor,third_obj_tensor],[first_z_where,second_z_where,third_z_where])
    save_image(gen_x,count,'gen',path='image/')

def save_example_image():
    train_iter = iter(train_loader)
    (data,_) = train_iter.next()
    print(type(data[0]))
    img = data.cpu().numpy().reshape(batch_size, A, B)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.savefig('image/example.png')

if __name__ == '__main__':
    save_example_image()
    train()
