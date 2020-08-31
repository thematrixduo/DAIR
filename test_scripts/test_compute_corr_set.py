import numpy as np
import itertools

def permute_c(c,perm):
    #print(type(c),c.dtype)
    permuted_c = np.zeros_like(c)
    permuted_c = perm[c.astype(np.int)]
    #for i in range(c.shape[0]):
    #    permuted_c[i]=perm[c[i]]
    return permuted_c

def corr_rate(a,b):
    a.sort(axis=1)
    #print(a)
    #print('--------------')
    b.sort(axis=1)
    #print(b)
    #print('--------------')
    dif = np.sum(a-b,axis=1)
    #print(dif)    
    #print('================')
    return 1 - np.mean(np.clip(dif,0,1))

def compute_correspondence(discrete_z, z_pres, c, c_mask):
    #print(discrete_z.shape,c.shape)
    z_idx = np.argmax(discrete_z,axis=-1)
    num_pred = np.sum(z_pres)
    #print(num_pred)
    c = c.astype(np.int)
    z_idx_masked = np.multiply(z_idx,z_pres) + z_pres - 1
    #print(discrete_z)
    #print(z_idx)
    #print(z_pres)
    #print(z_idx_masked)
    #print(c)
    #print(c_mask)
    #c_masked = np.multiply(c,z_pres)
    #print(z_pres)
    #print(z_idx_masked)
    #print(c_masked)    
    best_score = 0
    best_perm = np.zeros(3)
    for perm in itertools.permutations([0,1,2]):
        print(perm)
        permuted_c = permute_c(c,np.array(perm))
        c_masked = np.multiply(permuted_c,c_mask+1) + c_mask

        score = corr_rate(z_idx_masked,c_masked)
        print(score)
        if score > best_score:
            best_score = score
            best_perm = perm
            #print(z_idx_masked)
            #print(c_masked)


    print('corresponding classes:',best_perm)
    print('coorespondence rate= ',best_score,best_score,num_pred) 
    return best_score

acc_history = np.load('acc_history/acc_history_20000.npz')
discrete_z = acc_history['z_cat']
z_pres = acc_history['z_pres']
c = acc_history['labels']
c_mask = acc_history['labels_mask']

#print('pres:',z_pres[:10])
#print('cmask:',c_mask[:10])
#print('c:',c[:10])
#print('pred:',np.argmax(discrete_z[:10],axis=-1))

compute_correspondence(discrete_z,z_pres,c,c_mask)


