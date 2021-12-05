from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, update_average)
from torchvision.utils import save_image, make_grid
import os
import torch
from im2scene.training import BaseTrainer
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
logger_py = logging.getLogger(__name__)

is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )
    return conv

def addPadding(srcShapeTensor, tensor_whose_shape_isTobechanged):

    if(srcShapeTensor.shape != tensor_whose_shape_isTobechanged.shape):
        target = torch.zeros(srcShapeTensor.shape)
        target[:, :, :tensor_whose_shape_isTobechanged.shape[2],
               :tensor_whose_shape_isTobechanged.shape[3]] = tensor_whose_shape_isTobechanged
        return target
    return tensor_whose_shape_isTobechanged

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, image):
        # expected size
        # encoder (Normal convolutions decrease the size)
        x1 = self.down_conv_1(image)
        # print("x1 "+str(x1.shape))
        x2 = self.max_pool_2x2(x1)
        # print("x2 "+str(x2.shape))
        x3 = self.down_conv_2(x2)
        # print("x3 "+str(x3.shape))
        x4 = self.max_pool_2x2(x3)
        # print("x4 "+str(x4.shape))
        x5 = self.down_conv_3(x4)
        # print("x5 "+str(x5.shape))
        x6 = self.max_pool_2x2(x5)
        # print("x6 "+str(x6.shape))
        x7 = self.down_conv_4(x6)
        # print("x7 "+str(x7.shape))
        x8 = self.max_pool_2x2(x7)
        # print("x8 "+str(x8.shape))
        x9 = self.down_conv_5(x8)
        # print("x9 "+str(x9.shape))

        # decoder (transposed convolutions increase the size)
        x = self.up_trans_1(x9)
        x = addPadding(x7, x)
        x = self.up_conv_1(torch.cat([x7, x], 1))

        x = self.up_trans_2(x)
        x = addPadding(x5, x)
        x = self.up_conv_2(torch.cat([x5, x], 1))

        x = self.up_trans_3(x)
        x = addPadding(x3, x)
        x = self.up_conv_3(torch.cat([x3, x], 1))

        x = self.up_trans_4(x)
        x = addPadding(x1, x)
        x = self.up_conv_4(torch.cat([x1, x], 1))

        x = self.out(x)
        # print(x.shape)
        return x.to(device)

class Trainer(BaseTrainer):
    ''' Trainer object for GIRAFFE.

    Args:
        model (nn.Module): GIRAFFE model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''
    
    def __init__(self, model, optimizer, optimizer_d, device=None,
                 vis_dir=None,
                 multi_gpu=False, fid_dict={},
                 n_eval_iterations=10,
                 overwrite_visualization=True, **kwargs):
        print('Creando trainer 1')
        self.model = model
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir
        self.multi_gpu = multi_gpu
        self.mask_up = nn.Upsample(scale_factor=4.,mode='bilinear')


        self.overwrite_visualization = overwrite_visualization
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations

        self.vis_dict = model.generator.get_vis_dict(16)

        #self.segmentator = UNet().to(device)
        #self.segmentator.load_state_dict(torch.load('./model.pth')["model_state_dict"])
        #self.segmentator.eval()
        #toggle_grad(self.segmentator, False)


        self.segm_loss = nn.L1Loss()
        #self.umbral = torch.tensor([0.5],requires_grad=True,device='cuda')
        #self.u_opt = torch.optim.Adam([self.umbral], lr=0.0001)

        if multi_gpu:
            #self.segmentator = torch.nn.DataParallel(self.segmentator)
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(
                self.model.discriminator)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator
            self.generator_test = self.model.generator_test

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        loss_g = self.train_step_generator(data, it)
        loss_d, reg_d, fake_d, real_d = self.train_step_discriminator(data, it)

        return {
            'generator': loss_g,
            'discriminator': loss_d,
            'regularizer': reg_d,
        }
    def freeze_neural_renderer(self):
        self.generator.freeze_neural_renderer()


    def eval_step(self):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''

        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        x_fake = []
        n_iter = self.n_eval_iterations

        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                x_fake.append(gen().cpu()[:, :3])
        x_fake = torch.cat(x_fake, dim=0)
        x_fake.clamp_(0., 1.)
        mu, sigma = calculate_activation_statistics(x_fake)
        fid_score = calculate_frechet_distance(
            mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)
        eval_dict = {
            'fid_score': fid_score
        }

        return eval_dict

    def train_step_generator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator

        toggle_grad(generator, True)
        toggle_grad(discriminator, False)
        generator.train()
        discriminator.train()
        #alfa = False
        self.optimizer.zero_grad()
        #self.u_opt.zero_grad()
        if self.multi_gpu:
            latents = generator.module.get_vis_dict(batch_size=16)
            #x_fake = generator(**latents)
            #if alfa:
            x_fake, mask = generator(**latents,return_alpha_map=True,not_render_background=False)
            mask = self.mask_up(mask)
            #mask[mask>self.umbral] = 1
            x_fake_no_bg = generator(**latents,not_render_background=True)
            x_fake_bg = generator(**latents,only_render_background=True)
            x_fake_parts = x_fake_no_bg*mask+(1-mask)*x_fake_bg
            #print(x_fake.size(),x_fake_parts.size())
        else:
            x_fake = generator()
            #if alfa:
            #    x_fake_no_bg = generator(return_alpha_map=True,not_render_background=True)
            #else:
            #    x_fake_no_bg = generator(not_render_background=True)

        #if alfa:
        #    segment = self.segmentator(x_fake)
        #else: 
        #    segment = self.segmentator(x_fake)*x_fake

        
        d_fake = discriminator(x_fake)
        gloss = compute_bce(d_fake, 1)
        d_fake_parts = discriminator(x_fake_parts)
        gloss_parts = compute_bce(d_fake_parts, 1)
        l1loss = self.segm_loss(x_fake_parts,x_fake)
        tot_loss = 0*gloss+l1loss+gloss_parts
        tot_loss.backward()
        self.optimizer.step()
        #self.u_opt.step()
        #print(self.umbral)

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)

        return tot_loss.item()

    def train_step_discriminator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator
        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        generator.train()
        discriminator.train()

        self.optimizer_d.zero_grad()

        x_real = data.get('image').to(self.device)
        loss_d_full = 0.

        x_real.requires_grad_()
        d_real = discriminator(x_real)

        d_loss_real = compute_bce(d_real, 1)
        loss_d_full += d_loss_real

        reg = 10. * compute_grad2(d_real, x_real).mean()
        loss_d_full += reg

        with torch.no_grad():
            if self.multi_gpu:
                latents = generator.module.get_vis_dict()
                x_fake = generator(**latents)
            else:
                x_fake = generator()

        x_fake.requires_grad_()
        d_fake = discriminator(x_fake)

        d_loss_fake = compute_bce(d_fake, 0)
        loss_d_full += d_loss_fake

        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_fake + d_loss_real)

        return (
            d_loss.item(), reg.item(), d_loss_fake.item(), d_loss_real.item())

    def visualize(self, it=0):
        ''' Visualized the data.

        Args:
            it (int): training iteration
        '''
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        
        gen.eval()
        '''
        with torch.no_grad():
            image_fake = self.generator(**self.vis_dict, mode='val').cpu()
        '''
        generator = self.generator
        latents = generator.module.get_vis_dict(batch_size=16)
        x_fake, mask = generator(**latents,return_alpha_map=True,not_render_background=False)
        mask = self.mask_up(mask)
        #mask[mask>self.umbral] = 1
        x_fake_no_bg = generator(**latents,not_render_background=True)
        x_fake_bg = generator(**latents,only_render_background=True)
        x_fake_parts = x_fake_no_bg*mask+(1-mask)*x_fake_bg
        mask = torch.cat([mask,mask,mask],1)
        
    
        if self.overwrite_visualization:
            out_file_name = 'visualization.png' 
        else:
            out_file_name = 'visualization_%010d.png' % it
        image_fake = torch.cat([x_fake[:4].cpu(),mask[:4].cpu(),x_fake_no_bg.cpu()[:4],x_fake_parts.cpu()[:4],],0)
        #image_fake = x_fake
        image_grid = make_grid(image_fake.clamp_(0., 1.), nrow=4)
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        return image_grid,os.path.join(self.vis_dir, out_file_name)

class Trainer2(BaseTrainer):
    ''' Trainer object for GIRAFFE.

    Args:
        model (nn.Module): GIRAFFE model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''
    
    def __init__(self, model, optimizer, optimizer_d, device=None,
                 vis_dir=None,
                 multi_gpu=False, fid_dict={},
                 n_eval_iterations=10,
                 overwrite_visualization=True, **kwargs):
        print('Creando trainer 2')
        self.model = model
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir
        self.multi_gpu = multi_gpu

        self.overwrite_visualization = overwrite_visualization
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations

        self.segmentator = UNet().to(device)
        self.segmentator.load_state_dict(torch.load('./model.pth')["model_state_dict"])
        self.segmentator.eval()
        toggle_grad(self.segmentator, False)

        self.vis_dict = model.generator.get_vis_dict(16)

        if multi_gpu:
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(
                self.model.discriminator2)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator2
            self.generator_test = self.model.generator_test

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            
    def freeze_neural_renderer(self):
        self.generator.freeze_neural_renderer()

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        loss_g = self.train_step_generator(data, it)
        loss_d, reg_d, fake_d, real_d = self.train_step_discriminator(data, it)

        return {
            'generator': loss_g,
            'discriminator': loss_d,
            'regularizer': reg_d,
        }

    def eval_step(self):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''

        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        x_fake = []
        n_iter = self.n_eval_iterations

        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                x_fake.append(gen().cpu()[:, :3])
        x_fake = torch.cat(x_fake, dim=0)
        x_fake.clamp_(0., 1.)
        mu, sigma = calculate_activation_statistics(x_fake)
        fid_score = calculate_frechet_distance(
            mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)
        eval_dict = {
            'fid_score': fid_score
        }

        return eval_dict

    def train_step_generator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator

        toggle_grad(generator, True)
        toggle_grad(discriminator, False)
        generator.train()
        discriminator.train()

        self.optimizer.zero_grad()

        if self.multi_gpu:
            latents = generator.module.get_vis_dict()
            x_fake = generator(**latents,not_render_background=True)
        else:
            x_fake = generator(not_render_background=True)

        d_fake = discriminator(x_fake)
        gloss = compute_bce(d_fake, 1)

        gloss.backward()
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)

        return gloss.item()

    def train_step_discriminator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator
        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        generator.train()
        discriminator.train()

        self.optimizer_d.zero_grad()

        x_real = data.get('image').to(self.device)
        x_real = self.segmentator(x_real)*x_real
        
        loss_d_full = 0.

        x_real.requires_grad_()
        d_real = discriminator(x_real)

        d_loss_real = compute_bce(d_real, 1)
        loss_d_full += d_loss_real

        reg = 10. * compute_grad2(d_real, x_real).mean()
        loss_d_full += reg

        with torch.no_grad():
            if self.multi_gpu:
                latents = generator.module.get_vis_dict()
                x_fake = generator(**latents,not_render_background=True)
            else:
                x_fake = generator(not_render_background=True)

        x_fake.requires_grad_()
        d_fake = discriminator(x_fake)

        d_loss_fake = compute_bce(d_fake, 0)
        loss_d_full += d_loss_fake

        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_fake + d_loss_real)

        return (
            d_loss.item(), reg.item(), d_loss_fake.item(), d_loss_real.item())

    def visualize(self, it=0):
        ''' Visualized the data.

        Args:
            it (int): training iteration
        '''
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        with torch.no_grad():
            image_fake = self.generator(**self.vis_dict, mode='val').cpu()

        if self.overwrite_visualization:
            out_file_name = 'no_bg_visualization.png'
        else:
            out_file_name = 'no_bg_visualization_%010d.png' % it

        image_grid = make_grid(image_fake.clamp_(0., 1.), nrow=4)
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        return image_grid,os.path.join(self.vis_dir, out_file_name)

