import numpy as np
import itertools




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
    #print(discrete_z)
    print(z_idx)
    print(z_pres)
    print(z_idx_masked)
    #print(c)
    #print(c_mask)
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
            print(z_idx_masked)
            print(c_masked)


    print('corresponding classes:',best_perm)
    print('coorespondence rate= ',best_score/num_pred,best_score,num_pred) 
    return best_score/num_pred

acc_history = np.load('acc_history/acc_history_210000.npz')
discrete_z = acc_history['z_cat']
z_pres = acc_history['z_pres']
c = acc_history['labels']
c_mask = acc_history['labels_mask']

print(z_pres[:20])
print(c_mask[:20])


#compute_correspondence(discrete_z[:10],z_pres[:10],c[:10],c_mask[:10])
