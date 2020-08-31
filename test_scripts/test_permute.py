import torch
import numpy as np

torch_array = []
np_array = []
for i in range(3):
    data = np.random.normal(0,1,(64,3))
    #print(data.shape)
    np_array.append(data)
    torch_array.append(torch.tensor(data))

stacked_tensor = torch.stack(torch_array)
print(stacked_tensor.size())
permuted_tensor = stacked_tensor.permute(1,0,2)
print(permuted_tensor.size())
#print((stacked_tensor[0]-permuted_tensor[:,0,:]).data)
permuted_np = permuted_tensor.numpy()
print(np_array[0] - permuted_np[:,0,:])
