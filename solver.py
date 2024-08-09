from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

from color_dlib import image_lab2bgr, lipstick_color

# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure
from tqdm import tqdm



class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, mt_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader
        self.mt_loader = mt_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_bkg = config.lambda_bkg

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.select_device()

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()
        torch.manual_seed(100)


    def select_device(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: ", self.device)

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'MT', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        # print(name)
        # print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list
            

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)
        
    def color_distance_loss(self, logit, target):
        """Compute MSE loss of LAB color space."""
        return F.mse_loss(logit, target)
    
    def mssim_loss(self, logit, target):
        """Compute multiscale structural similarity loss of the generative image."""
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(kernel_size=5).to(self.device)
        return 1 - ms_ssim( logit, target)
        # return 1 - ms_ssim( logit, target, data_range=255, size_average=True )
    
    def ssim_loss(self, logit, target):
        """Compute multiscale structural similarity loss of the generative image."""
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        return 1 - ssim( logit, target)
        # return 1 - ms_ssim( logit, target, data_range=255, size_average=True )

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'MT':
            data_loader = self.mt_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_fixed = next(data_iter)
        x_target, c_target = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed = c_fixed.to(self.device)
        x_target = x_target.to(self.device)
        c_target = c_target.to(self.device)

        label_trg = c_fixed
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in tqdm(range(start_iters, self.num_iters)):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # make sure the size of label_trg == label_org
            if len(label_trg) > len(label_org):
                label_trg = label_trg[:len(label_org)]
            elif len(label_trg) < len(label_org):
                label_org = label_org[:len(label_trg)]
                x_real = x_real[:len(label_trg)]
                    

            # # Generate target domain labels randomly.
            # rand_idx = torch.randperm(label_org.size(0)) # for random [0,n]
            # label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
            elif self.dataset == 'MT':
                c_org = label_org.clone()
                c_trg = label_trg.clone()

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls, out_cls_1 = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.color_distance_loss(out_cls, label_org[:, 0, :]) # label[0] = lips lab color
            d_loss_cls_1 = self.color_distance_loss(out_cls_1, label_org[:, 1, :]) # label[1] = skin lab color
            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg[:, 0, :])
            out_src, out_cls, out_cls_1 = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + d_loss_cls + d_loss_cls_1 + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg[:, 0, :])
                out_src, out_cls, out_cls_1 = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.color_distance_loss(out_cls, label_trg[:, 0, :]) # label[0] = lips lab color
                g_loss_cls_1 = self.color_distance_loss(out_cls_1, label_org[:, 1, :]) # label[1] = skin lab color

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org[:, 0, :])
                # g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                g_loss_rec = self.ssim_loss(x_real, x_reconst)

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + self.lambda_bkg * g_loss_cls_1
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    # create color strip for original lips
                    lip_color_org_list = []
                    for index,lip_color in enumerate(c_fixed[:, 0, :]):
                        color_image = torch.zeros_like(x_fixed[index])
                        color_image[:] = lip_color[:, None, None]
                        color_image = image_lab2bgr(color_image.cpu().numpy(), lip_color.cpu().numpy())
                        lip_color_org_list.append(torch.from_numpy(color_image))
                    lip_color_org_tensor = torch.stack(lip_color_org_list)
                    # create color strip for target lips
                    lip_color_trg_list = []
                    for index,lip_color in enumerate(c_target[:, 0, :]):
                        color_image = torch.zeros_like(x_target[index])
                        color_image[:] = lip_color[:, None, None]
                        color_image = image_lab2bgr(color_image.cpu().numpy(), lip_color.cpu().numpy())
                        lip_color_trg_list.append(torch.from_numpy(color_image))
                    lip_color_trg_tensor = torch.stack(lip_color_trg_list)
                    x_list = []
                    # original image
                    original = self.denorm(x_fixed.data.cpu())
                    # generated image
                    fake = self.denorm(self.G(x_fixed, c_target[:, 0, :]).data.cpu())
                    # target image
                    target = self.denorm(x_target.data.cpu())
                    # original lips color
                    original_color = lip_color_org_tensor / 255.
                    # target lips color
                    target_color = lip_color_trg_tensor / 255.
                    # append all images to x_list
                    x_list.append(original_color)
                    x_list.append(original)
                    x_list.append(fake)
                    x_list.append(target)
                    x_list.append(target_color)
                    x_concat = torch.cat(x_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(x_concat, sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))


            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            
            # prevent memory leak
            if (i+1) % 20000 == 0:
                torch.cuda.empty_cache()
                print('Memory Cleared')
            # Update target color
            label_trg = label_org    

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        elif self.dataset == 'MT':
            data_loader = self.mt_loader

        # Monk skin tone 1 to 10
        # test_file_names = ['vRX447.png','vRX490.png','vHX478.png','vFG58.png','vHX120.png','vFG333.png','XYH-082.png','8fb420459611cbe7e1e87f04abaa505f.png','Mypsd_2969_201012102201250011B.png','32-41039.png']
        color_pH = ['55','60','65','70','80']
        for c in range(4):
            test_lips_color = torch.tensor(lipstick_color(c))
            test_lips_color = test_lips_color.float()
            with torch.no_grad():

                for i, (x_real, c_org, filename, skintone) in enumerate(data_loader):
                    # create color strip for original lips
                    x_fixed = x_real.to(self.device)
                    c_fixed = c_org.to(self.device)
                    test_lips_color = test_lips_color.to(self.device)
                    
                    lip_color_org_list = []
                    for index, lip_color in enumerate(c_fixed[:, 0, :]):
                        color_image = torch.zeros_like(x_fixed[index])
                        color_image[:] = lip_color[:, None, None]
                        color_image = image_lab2bgr(color_image.cpu().numpy(), lip_color.cpu().numpy())
                        lip_color_org_list.append(torch.from_numpy(color_image))
                    lip_color_org_tensor = torch.stack(lip_color_org_list)

                    # create color strip for target lips multiple shades
                    color_images = [torch.zeros_like(x_fixed[0]).cpu(),torch.zeros_like(x_fixed[0]).cpu()]
                    for lip_color in test_lips_color:
                        color_image = torch.zeros_like(x_fixed[0])
                        lip_color = lip_color.resize(3, 1, 1)
                        color_image[:] = lip_color
                        color_image = image_lab2bgr(color_image.cpu().numpy(), lip_color.cpu().numpy())
                        color_images.append(torch.from_numpy(color_image))
                    color_concat = torch.cat(color_images, dim=-1)
                    color_concat = color_concat.unsqueeze(0)

                    x_list = []
                    # original image
                    original = self.denorm(x_fixed.data.cpu())
                    # generated image
                    fake_list = []
                    for index, lip_color in enumerate(test_lips_color):
                        color_batches = lip_color.repeat(len(filename), 1)
                        fake = self.denorm(self.G(x_fixed, color_batches).data.cpu())
                        fake_list.append(fake)
                        # save images
                        # save_image(fake, sample_path, nrow=1, padding=0)
                        for j, image in enumerate(fake):
                            sample_path = os.path.join(self.sample_dir, '{}-{}-{}-{}.jpg'.format(skintone[j],color_pH[index],c,filename[j].split('.')[0]))
                            save_image(image, sample_path, padding=0)
                    fake_concat = torch.cat(fake_list, dim=-1)
                    # original lips color
                    original_color = lip_color_org_tensor / 255.
                    # # target lips color (single shade)
                    # target_color_single = lip_color_trg_tensor_single / 255.
                    color_concat = color_concat / 255.

                    # append all images to x_list
                    x_list.append(original_color)
                    x_list.append(original)
                    x_list.append(fake_concat)
                    # x_list.append(target_color_single)
                    x_concat = torch.cat(x_list, dim=3)
                    
                    final = torch.cat((color_concat, x_concat), dim=0)
                    sample_path = os.path.join(self.sample_dir, '{}-{}_images.jpg'.format(i+1, c))
                    save_image(final, sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))
