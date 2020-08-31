import torch
import torch.nn as nn
from utility import *
import torch.nn.functional as F
from gumbel_sigmoid import gumbel_sigmoid
import numpy as np
import math

class DrawModel(nn.Module):
    def __init__(self,T,A,B,z_size,cat_size,window_size,rnn_size,device='cuda',
        scale_mu_prior=4, scale_sigma_prior=0.3,pos_mu_prior=0.0,pos_sigma_prior=3.0):
        super(DrawModel,self).__init__()
        self.T = T
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        self.z_size = z_size
        self.cat_size = cat_size
        self.window_size = window_size
        self.device=device
        self.cat_prior = math.log(1.0/self.cat_size)
        #self.dec_size = dec_size
        self.rnn_size = rnn_size
        self.cs = [0] * T
        self.cats = [0] * T
        self.logsigmas,self.sigmas,self.mus = [0] * T,[0] * T,[0] * T
        self.z_where_logsigmas,self.z_where_sigmas,self.z_where_mus = [0] * T,[0] * T,[0] * T
        self.z_where_hist = [0] * T
        self.z_gumbel_hist = [0] * T
        self.gumbel_pre_sigmoid_hist = [0] * T
        self.z_stop_hist = [0] * T
        self.encoder_fc_1 = nn.Linear(96*4*4,256)
        self.enc_fc_1_bn = nn.BatchNorm1d(256)
        self.encoder_fc_2 = nn.Linear(256,128)
        self.enc_fc_2_bn = nn.BatchNorm1d(128)
        #self.encoder_fc1 = nn.Linear(self.window_size **2,200)
        #self.decoder_fc1 = nn.Linear(50,200)
        #self.decoder_fc2 = nn.Linear(200,self.window_size **2) 
        self.decoder_fc = nn.Linear(cat_size,64*4*4)
        self.dec_fc_bn = nn.BatchNorm1d(64*4*4)

        self.scale_mu_prior = scale_mu_prior
        self.pos_mu_prior = pos_mu_prior
        self.scale_sigma_prior = scale_sigma_prior
        self.pos_sigma_prior = pos_sigma_prior
        #self.convOut = nn.Conv2d(4, 1, kernel_size=1)

        self.RNN = nn.LSTMCell(32*8*8+cat_size+3+1,rnn_size)
        self.loc_hidden_fc = nn.Linear(rnn_size,200)

        self.cat_linear = nn.Linear(128,cat_size)

        self.z_stop_linear = nn.Linear(200, 1)
        self.z_where_mu_linear = nn.Linear(200, 3)
        self.z_where_sigma_linear = nn.Linear(200, 3)
        self.z_where_mu_linear.bias.data.copy_(torch.tensor([2, 0, 0], dtype=torch.float))
        self.z_where_sigma_linear.bias.data.copy_(torch.tensor([0.3, 1, 1], dtype=torch.float))
        
        self.rot_angle_linear = nn.Linear(32,1)
        self.rot_angle_linear.weight.data.zero_()
        self.rot_angle_linear.bias.data.copy_(torch.tensor([0], dtype=torch.float))

        self.conv_enc = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(True),            

            nn.Conv2d(48, 64, kernel_size=5,stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 96, kernel_size=5,stride=2, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(True)

        )

        self.conv_dec = nn.Sequential(
            nn.ConvTranspose2d(64, 48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(True), 

            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),            

            nn.ConvTranspose2d(32, 2, kernel_size=4,stride=2, padding=1),
            #nn.Sigmoid()

        )

        #Spatial Transformer Layers
        self.localization_enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(True),            
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 24, kernel_size=5, padding=2),
            #nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(24, 32, kernel_size=5, padding=2), 
            #nn.BatchNorm2d(32),           
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)

        )




        self.localization_rotate = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(True),            
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),

        )


        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_rotate = nn.Sequential(
            nn.Linear(64 * 5 * 5, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_theta_enc = nn.Linear(32,6)
        self.fc_theta_enc.weight.data.zero_()
        self.fc_theta_enc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc_theta_dec = nn.Linear(32,6)
        self.fc_theta_dec.weight.data.zero_()
        self.fc_theta_dec.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


        self.sigmoid = nn.Sigmoid()

    def expand_z_where(self, z_where):
        # Takes 3-dimensional vectors, and massages them into 2x3 matrices with elements like so:
        # [s,x,y] -> [[s,0,x],
        #             [0,s,y]]
        n = z_where.size(0)
        out = torch.cat((z_where.new_zeros(n, 1), z_where), 1)
        ix = torch.LongTensor([1, 0, 2, 0, 1, 3])
        if z_where.is_cuda:
            ix = ix.cuda()
        out = torch.index_select(out, 1, ix)
        out = out.view(n, 2, 3)
        return out

    def z_where_inv(self, z_where):
        # Take a batch of z_where vectors, and compute their "inverse".
        # That is, for each row compute:
        # [s,x,y] -> [1/s,-x/s,-y/s]
        # These are the parameters required to perform the inverse of the
        # spatial transform performed in the generative model.
        n = z_where.size(0)
        out = torch.cat((torch.ones([1, 1]).type_as(z_where).expand(n, 1), -z_where[:, 1:]), 1)
        out = out / z_where[:, 0:1]
        return out

    def read(self, x, z_where):
        n = x.size(0)
        theta_inv = self.expand_z_where(self.z_where_inv(z_where))
        grid = F.affine_grid(theta_inv, torch.Size((n, 1, self.window_size, self.window_size)))
        out = F.grid_sample(x.view(n, 1, A, B), grid)
        return out#.view(n, -1)

    def write(self, obj, z_where):
        n = obj.size(0)
        theta = self.expand_z_where(z_where)
        grid = F.affine_grid(theta, torch.Size((n, 1, A, B)))
        #print(obj.size())
        out = F.grid_sample(obj.view(n, 1, self.window_size, self.window_size), grid)
        return out.view(n, 1, A, B)

    def rotate(self,x_st):
        zeros_array = torch.zeros(self.batch_size,1).to(self.device) 
        loc_fmap = self.localization_rotate(x_st)
        loc_feature = self.fc_loc_rotate(loc_fmap.view(-1,64*5*5))
        rot_angle= F.tanh(self.rot_angle_linear(loc_feature))
        cos_rot = torch.cos(torch.mul(rot_angle,np.pi))
        sin_rot = torch.sin(torch.mul(rot_angle,np.pi))
        #print(zeros_array.size(),cos_rot.size(),sin_rot.size())

        rot_mat = torch.cat((cos_rot,sin_rot,zeros_array,-sin_rot,cos_rot,zeros_array),1).view(-1,2,3)
        rot_mat_inv = torch.cat((cos_rot,-sin_rot,zeros_array,sin_rot,cos_rot,zeros_array),1).view(-1,2,3)
        grid = F.affine_grid(rot_mat, x_st.size())
        x_str = F.grid_sample(x_st, grid)
        #theta_dec = torch.matmul(rot_mat,trans_scale_mat_dec)
        return x_str,rot_mat_inv

    def dec_rotate(self, x, rot_mat):

        grid = F.affine_grid(rot_mat, x.size())
        x = F.grid_sample(x, grid)

        return x


    def normalSample(self,sample_size):
        return Variable(torch.randn(self.batch_size,sample_size))

    # correct
    def compute_mu(self,g,rng,delta):
        rng_t,delta_t = align(rng,delta)
        tmp = (rng_t - self.N / 2 - 0.5) * delta_t
        tmp_t,g_t = align(tmp,g)
        mu = tmp_t + g_t
        return mu

    def forward(self,x,tau):
        self.batch_size = x.size()[0]
        h_prev = Variable(torch.zeros(self.batch_size, self.rnn_size))
        z_gumbel_prev = Variable(torch.ones(self.batch_size,1))
        rnn_state = Variable(torch.zeros(self.batch_size,self.rnn_size))
        #z_cat_prev = Variable(torch.zeros(self.batch_size,self.cat_size))
        z_where_prev = Variable(torch.zeros(self.batch_size,3))
        
        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size, 1 ,self.A , self.B)) if t == 0 else self.cs[t-1]
            z_cat_prev = Variable(torch.zeros(self.batch_size,self.cat_size)) if t==0 else self.cats[t-1]
            #print(x.size())
            x_hat = x - self.sigmoid(c_prev)
            loc_feature = self.localization_enc(x_hat)
            rnn_input = torch.cat((loc_feature.view(-1,32*8*8),z_cat_prev,z_where_prev,z_gumbel_prev),1)
            
            h_prev,rnn_state = self.RNN(rnn_input,(h_prev,rnn_state))
            
            loc_hidden = F.relu(self.loc_hidden_fc(h_prev))
            z_where, self.z_where_mus[t], self.z_where_logsigmas[t], self.z_where_sigmas[t], z_gumbel, self.z_stop_hist[t], self.gumbel_pre_sigmoid_hist[t] = self.sample_where_prez(loc_hidden,z_gumbel_prev, tau)
            self.z_gumbel_hist[t] = z_gumbel
            self.z_where_hist[t] = z_where
            x_st = self.read(x,z_where)
            #print(z_gumbel.data)
            #print(x_st.size())
            #enc_feature = self.encoder(x_st)
            x_st, rot_mat_inv = self.rotate(x_st)
            enc_fmap = self.conv_enc(x_st).view(-1,96*4*4)
            enc_feature = F.relu(self.enc_fc_1_bn(self.encoder_fc_1(enc_fmap)))
            enc_feature = F.relu(self.enc_fc_2_bn(self.encoder_fc_2(enc_feature)))

            #enc_feature = F.relu(self.encoder_fc(enc_fmap))
            #z,self.mus[t],self.logsigmas[t],self.sigmas[t], self.cats[t] = self.sampleQ(enc_feature,tau)
            self.cats[t] = self.sample_cat(enc_feature,tau)
            dec_fc = F.relu(self.dec_fc_bn(self.decoder_fc(self.cats[t])))
            #dec_fc = F.relu(self.decoder_fc(self.cats[t]))
            #x_dec = self.decoder(z)
            #print(dec_fc.view(-1,64,4,4).size())
            conv_dec_out = self.conv_dec(dec_fc.view(-1,64,4,4))
            x_dec = F.sigmoid(conv_dec_out[:,0:1,:,:]-2)
            x_dec = self.dec_rotate(x_dec,rot_mat_inv)
            out_canvas = self.write(x_dec,z_where)

            scale_obj,_ = gumbel_sigmoid(conv_dec_out[:,1:2,:,:])
            scale_obj = self.dec_rotate(scale_obj,rot_mat_inv)
            scale_out = self.write(scale_obj,z_where)
            scale_out = scale_out * z_gumbel.view(-1,1,1,1)

            #print('x_dec: ',torch.min(x_dec),torch.max(x_dec))

            #print('out_canvas: ',torch.min(out_canvas),torch.max(out_canvas))
            #print(out_canvas.size())
            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = c_prev + out_canvas * scale_out
            #print(self.cs[t])
            #print('cs[t]: ',torch.min(self.cs[t]),torch.max(self.cs[t]))
            z_gumbel_prev = z_gumbel
            z_where_prev = z_where

        return self.z_where_hist, self.z_gumbel_hist, self.cats


    def loss(self,x,tau, z_pres_prior_p,eps=1e-8):
        _,_,_ = self.forward(x,tau)
        criterion = nn.MSELoss(size_average = False)
        x_recons = self.cs[-1]
        #print(type(self.cs[-1]),type(self.cs[-2]))
        #print(torch.max(x_recons))
        #print(type(x_recons),type(x))
        Lx = criterion(x_recons,x)# * self.A * self.B
        Lz_what = 0
        what_kl_terms = [0] * T
        for t in range(self.T):
            what_kl_terms[t] = torch.sum(self.cats[t]*(torch.log(self.cats[t]+eps)-self.cat_prior),1)
            Lz_what += torch.round(self.z_gumbel_hist[t]) * what_kl_terms[t]
        
        Lz_where = 0
        where_kl_terms = [0] * T
        where_mu_prior = torch.tensor([self.scale_mu_prior, self.pos_mu_prior, self.pos_mu_prior]).cuda()
        where_sigma_prior = torch.tensor([self.scale_sigma_prior, self.pos_sigma_prior, self.pos_sigma_prior]).cuda()

        for t in range(self.T):
            mu_2 = (self.z_where_mus[t]-where_mu_prior) * (self.z_where_mus[t]-where_mu_prior)
            sigma_2 = self.z_where_sigmas[t] * self.z_where_sigmas[t]
            logsigma = self.z_where_logsigmas[t]
            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))    # 11
            #print((where_sigma_prior*where_sigma_prior))
            where_kl_terms[t] = 0.5 * torch.sum((mu_2+sigma_2)/(where_sigma_prior * where_sigma_prior) + 2*torch.log(where_sigma_prior)-2 * logsigma,1) -  0.5
            Lz_where += torch.round(self.z_gumbel_hist[t]) * where_kl_terms[t]

        Lz_pres = 0
        pres_kl_terms = [0] * T
        for t in range(self.T):
            z_stop = self.z_stop_hist[t]
            y = self.gumbel_pre_sigmoid_hist[t]
            z_prior = z_pres_prior_p(t)
            
            #print(t,z_log_prior)
            pres_kl_terms[t] = z_stop - z_prior +2 * (torch.log(1 + torch.exp(z_prior-y) + eps)-torch.log(1+torch.exp(z_stop-y)+eps))

            #pres_kl_terms[t] = z_gumbel * (torch.log(z_gumbel+1e-7)-z_log_prior)+(1-z_gumbel)*(torch.log(1-z_gumbel+1e-7)-one_minus_z_log_prior)
            Lz_pres += pres_kl_terms[t]

        # Lz -= self.T / 2
        Lz_what = torch.mean(Lz_what)
        Lz_where = torch.mean(Lz_where)    ####################################################
        Lz_pres = torch.mean(Lz_pres)
        #print(Lx.data,Lz_what.data,Lz_where.data,Lz_pres.data)
        loss = Lx  +  Lz_where + 30 * Lz_pres   # 12
        return loss

    def get_where_pres(self):
        return (self.z_where_hist,self.z_gumbel_hist)

    '''
    def sampleQ(self,h_enc,tau):
        e = self.normalSample(self.z_size)
        # mu_sigma = self.mu_sigma_linear(h_enc)
        # mu = mu_sigma[:, :self.z_size]
        # log_sigma = mu_sigma[:, self.z_size:]
        mu = self.mu_linear(h_enc)           # 1
        log_sigma = self.sigma_linear(h_enc) # 2
        sigma = torch.exp(log_sigma)  
        cat = F.gumbel_softmax(self.cat_linear(h_enc),tau)
        return mu + sigma * e , mu , log_sigma, sigma, cat
    '''
    def sample_cat(self,h_enc,tau): 
        cat = F.gumbel_softmax(self.cat_linear(h_enc),tau)
        return cat

    def sample_where_prez(self,h, z_gumbel_prev, tau):
        e = self.normalSample(3)
        # mu_sigma = self.mu_sigma_linear(h_enc)
        # mu = mu_sigma[:, :self.z_size]
        # log_sigma = mu_sigma[:, self.z_size:]
        mu = self.z_where_mu_linear(h)
        log_sigma = self.z_where_sigma_linear(h)
        sigma = torch.exp(log_sigma) 
        z_stop = self.z_stop_linear(h) 
        #z_gumbel = gumbel_sigmoid(z_stop * z_gumbel_prev.detach() ,tau= tau)
        
        z_gumbel,gumbel_pre_sigmoid = gumbel_sigmoid(z_stop , tau= tau)
        z_gumbel = z_gumbel * z_gumbel_prev.detach()       
        
        return mu + sigma * e , mu , log_sigma, sigma, z_gumbel, z_stop, gumbel_pre_sigmoid

    def get_transform_mat(self,test_gen_size):

        transform_mat = np.zeros((test_gen_size,6))

        for i in range(8):
            print(i)
            transform_mat[i]=np.array([np.cos(i*3.14/48),np.sin(i*3.14/48),0,-np.sin(i*3.14/48),np.cos(i*3.14/48),0])
            transform_mat[8+i]=np.array([np.cos((8+i)*3.14/48),np.sin((8+i)*3.14/48),0,-np.sin((8+i)*3.14/48),np.cos((8+i)*3.14/48),0])
            transform_mat[16+i]=np.array([np.cos((16+i)*3.14/48),np.sin((16+i)*3.14/48),0,-np.sin((16+i)*3.14/48),np.cos((16+i)*3.14/48),0])
            transform_mat[24+i]=np.array([np.cos(i*3.14/48),-np.sin(i*3.14/48),0,np.sin(i*3.14/48),np.cos(i*3.14/48),0])
            transform_mat[32+i]=np.array([np.cos((8+i)*3.14/48),-np.sin((8+i)*3.14/48),0,np.sin((8+i)*3.14/48),np.cos((8+i)*3.14/48),0])
            transform_mat[40+i]=np.array([np.cos((16+i)*3.14/48),-np.sin((16+i)*3.14/48),0,np.sin((16+i)*3.14/48),np.cos((16+i)*3.14/48),0])
            transform_mat[48+i]=np.array([1,0,0,0,1,0])
            transform_mat[56+i]=np.array([1,0,0,0,1,0])

        return transform_mat

    def z_interpolate(self,start=-1.2,stop = 1.2, row_size=8, batch_size = 64):
        #print(type(start),type(stop),type(row_size))
        step_size = ( stop - start ) / (row_size-1) 
        z = np.zeros((batch_size,2))
        for i in range(batch_size):
            row = i // row_size
            col = i % row_size
            z[i,0] = start + row*step_size
            z[i,1] = start + col*step_size
        return torch.from_numpy(z).float().to(self.device)

    def generate(self,cats,z_where,batch_size=64):
        self.batch_size = batch_size
        h_prev = Variable(torch.zeros(self.batch_size,self.rnn_size),volatile = True)
        rnn_state = Variable(torch.zeros(self.batch_size, self.rnn_size),volatile = True)
        transform_mat = self.get_transform_mat(batch_size)
        #transform_mat = np.tile(np.array([1,0,0,0,1,0]),(batch_size,1))
        transform_mat_tensor = torch.from_numpy(transform_mat).float()
        transform_mat_tensor = transform_mat_tensor.view(-1, 2, 3).to(self.device)

        for t in range(self.T-1):
            c_prev = Variable(torch.zeros(self.batch_size, 1, self.A , self.B)) if t == 0 else self.cs[t - 1]
            #z = self.normalSample(self.z_size)

            dec_fc = F.relu(self.dec_fc_bn(self.decoder_fc(cats[t])))
            #dec_fc = F.relu(self.decoder_fc(self.cats[t]))
            #x_dec = self.decoder(z)
            #print(dec_fc.view(-1,64,4,4).size())
            conv_dec_out = self.conv_dec(dec_fc.view(-1,64,4,4))
            x_dec = F.sigmoid(conv_dec_out[:,0:1,:,:]-2)
            x_dec = self.dec_rotate(x_dec,transform_mat_tensor)
            out_canvas = self.write(x_dec,z_where[t])

            scale_obj,_ = gumbel_sigmoid(conv_dec_out[:,1:2,:,:])
            scale_obj = self.dec_rotate(scale_obj,transform_mat_tensor)
            scale_out = self.write(scale_obj,z_where[t])
            scale_out = scale_out 

            #print('x_dec: ',torch.min(x_dec),torch.max(x_dec))

            #print('out_canvas: ',torch.min(out_canvas),torch.max(out_canvas))
            #print(out_canvas.size())
            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = out_canvas * scale_out

        imgs = []
        for img in self.cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return imgs

    def recon(self,x,tau):
        _,z_pres,cats = self.forward(x,tau)
        #print(cats.data[:8])
        imgs = []
        for img in self.cs:
            imgs.append(img.cpu().data.numpy())
        return imgs,z_pres ,cats        



# model = DrawModel(10,5,5,10,5,128,128)
# x = Variable(torch.ones(4,25))
# x_hat = Variable(torch.ones(4,25)*2)
# r = model.write()
# print r
# g = Variable(torch.ones(4,1))
# delta = Variable(torch.ones(4,1)  * 3)
# sigma = Variable(torch.ones(4,1))
# rng = Variable(torch.arange(0,5).view(1,-1))
# mu_x = model.compute_mu(g,rng,delta)
# a = Variable(torch.arange(0,5).view(1,1,-1))
# mu_x = mu_x.view(-1,5,1)
# sigma = sigma.view(-1,1,1)
# F = model.filterbank_matrices(a,mu_x,sigma)
# print F
# def test_normalSample():
#     print model.normalSample()
#
# def test_write():
#     h_dec = Variable(torch.zeros(8,128))
#     model.write(h_dec)
#
# def test_read():
#     x = Variable(torch.zeros(8,28*28))
#     x_hat = Variable((torch.zeros(8,28*28)))
#     h_dec = Variable(torch.zeros(8, 128))
#     model.read(x,x_hat,h_dec)
#
# def test_loss():
#     x = Variable(torch.zeros(8,28*28))
#     loss = model.loss(x)
#     print loss

