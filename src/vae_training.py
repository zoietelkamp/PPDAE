import numpy as np
import datetime
import torch
import torch.nn as nn
from src.utils import plot_recon_wall, plot_latent_space
from src.training_callbacks import EarlyStopping


class Trainer():
    def __init__(self, model, optimizer, batch_size, wandb,
                 scheduler=None, print_every=50, beta='step',
                 device='cpu'):

        self.device = device
        self.model = model
        if torch.cuda.device_count() > 1 and True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        print('Is model in cuda? ', next(self.model.parameters()).is_cuda)
        self.opt = optimizer
        self.sch = scheduler
        self.batch_size = batch_size
        self.train_loss = {'Loss': [], 'MSE': [], 'KLD': []}
        self.test_loss = {'Loss': [], 'MSE': [], 'KLD': []}
        self.num_steps = 0
        self.print_every = print_every
        self.mse_loss = nn.MSELoss(reduction='mean')
        # self.kld_loss = nn.KLDivLoss(reduction='mean')
        self.wb = wandb
        self.beta = beta

    def _beta_scheduler(self, epoch, beta0=0., step=15, gamma=0.2):
        if self.beta == 'step':
            return beta0 + gamma * (epoch // step)
        else:
            return float(self.beta)

    def _loss(self, x, xhat, mu, logvar, train=True, ep=0):
        mse = self.mse_loss(xhat, x)
        kld_l = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + self._beta_scheduler(ep) * kld_l

        if train:
            self.train_loss['MSE'].append(mse.item())# / len(x))
            self.train_loss['KLD'].append(kld_l.item())# / len(x))
            self.train_loss['Loss'].append(loss.item())# / len(x))
        else:
            self.test_loss['MSE'].append(mse.item())# / len(x))
            self.test_loss['KLD'].append(kld_l.item())# / len(x))
            self.test_loss['Loss'].append(loss.item())# / len(x))

        return loss

    def _train_epoch(self, data_loader, epoch):
        self.model.train()
        # iterate over len(data)/batch_size
        z_all = []
        xhat_plot, x_plot = [], []
        for i, (img, meta) in enumerate(data_loader):
            self.num_steps += 1
            self.opt.zero_grad()
            img = img.to(self.device)

            xhat, z, mu, logvar = self.model(img)

            loss = self._loss(img, xhat, mu, logvar,
                              train=True, ep=epoch)
            loss.backward()
            self.opt.step()

            self._report_train(i)
            z_all.append(mu.data.cpu().numpy())
            if i == len(data_loader) - 1:
                xhat_plot = xhat.data.cpu().numpy()
                x_plot = img.data.cpu().numpy()

        z_all = np.concatenate(z_all)
        z_all = z_all[np.random.choice(z_all.shape[0], 1000,
                                       replace=False), :]

        if epoch % 1 == 0:
            wall = plot_recon_wall(xhat_plot, x_plot, epoch=epoch)
            self.wb.log({'Train_Recon':  self.wb.Image(wall)},
                        step=self.num_steps)

        if epoch % 1 == 0:
            latent_plot = plot_latent_space(z_all, y=None)
            self.wb.log({'Latent_space': self.wb.Image(latent_plot)},
                        step=self.num_steps)

    def _test_epoch(self, test_loader, epoch):
        self.model.eval()
        with torch.no_grad():
            xhat_plot, x_plot = [], []

            for i, (img, meta) in enumerate(test_loader):
                img = img.to(self.device)
                xhat, z, mu, logvar = self.model(img)
                loss = self._loss(img, xhat, mu, logvar,
                                  train=False, ep=epoch)

                if i == len(test_loader) - 1:
                    xhat_plot = xhat.data.cpu().numpy()
                    x_plot = img.data.cpu().numpy()

        self._report_test(epoch)

        # generate data with G for visualization and seve to tensorboard
        if epoch % 2 == 0:
            wall = plot_recon_wall(xhat_plot, x_plot, epoch=epoch)
            self.wb.log({'Test_Recon':  self.wb.Image(wall)},
                        step=self.num_steps)

        return loss

    def train(self, train_loader, test_loader, epochs,
              save=True, early_stop=False):

        # hold samples, real and generated, for initial plotting
        if early_stop:
            early_stopping = EarlyStopping(patience=10, min_delta=.1,
                                           verbose=True)

        # train for n number of epochs
        time_start = datetime.datetime.now()
        for epoch in range(1, epochs + 1):
            e_time = datetime.datetime.now()
            print('##'*20)
            print("\nEpoch {}".format(epoch))

            # train and validate
            self._train_epoch(train_loader, epoch)
            val_loss = self._test_epoch(test_loader, epoch)

            # update learning rate according to cheduler
            if self.sch is not None:
                self.wb.log({'LR': self.opt.param_groups[0]['lr']},
                            step=self.num_steps)
                if 'ReduceLROnPlateau' == self.sch.__class__.__name__:
                    self.sch.step(val_loss)
                else:
                    self.sch.step(epoch)

            # report elapsed time per epoch and total run tume
            epoch_time = datetime.datetime.now() - e_time
            elap_time = datetime.datetime.now() - time_start
            print('Time per epoch: ', epoch_time.seconds, ' s')
            print('Elapsed time  : %.2f m' % (elap_time.seconds/60))
            print('##'*20)

            # early stopping
            if early_stop:
                early_stopping(val_loss.cpu())
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if save:
            torch.save(self.model.state_dict(), '%s/model.pt' %
                       (self.wb.run.dir))

    def _report_train(self, i):
        # ------------------------ Reports ---------------------------- #
        # print scalars to std output and save scalars/hist to W&B
        if i % self.print_every == 0:
            print("Training iteration %i, global step %i" %
                  (i + 1, self.num_steps))
            print("Loss: %.4f" % (self.train_loss['Loss'][-1]))

            self.wb.log({'Train_Loss': self.train_loss['Loss'][-1],
                         'Train_MSE': self.train_loss['MSE'][-1],
                         'Train_KLD': self.train_loss['KLD'][-1]},
                        step=self.num_steps)
            print("__"*20)

    def _report_test(self, ep):
        # ------------------------ Reports ---------------------------- #
        # print scalars to std output and save scalars/hist to W&B
        print('*** TEST LOSS ***')
        print("Epoch %i, global step %i" % (ep, self.num_steps))
        print("Loss: %.4f" % (self.test_loss['Loss'][-1]))

        self.wb.log({'Test_Loss': self.test_loss['Loss'][-1],
                     'Test_MSE': self.test_loss['MSE'][-1],
                     'Test_KLD': self.test_loss['KLD'][-1]},
                    step=self.num_steps)
        print("__"*20)
