import os
import time

import numpy as np

from utils import format_time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from skimage.color import lab2rgb
import torchmetrics


class Checkpoint:

    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer, gen_scheduler,
                 disc_scheduler, path):
        """
        :param generator: Generator model
        :param discriminator: Discriminator model
        :param gen_optimizer: Generator optimizer
        :param disc_optimizer: Discriminator optimizer
        :param gen_scheduler: Generator lr scheduler
        :param disc_scheduler: Discriminator lr scheduler
        :param path: checkpoint file path
        """
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.path = path

    def load(self):
        """
        Loads the generator, discriminator, optimizers and schedulers states and history from the checkpoint file.
        :return: history if checkpoint exists, else None
        """
        if os.path.exists(self.path):
            print("Loading checkpoint from {}".format(self.path))
            checkpoint = torch.load(self.path)
            self.generator.load_state_dict(checkpoint['generator'])
            print("Generator loaded")
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            print("Discriminator loaded")
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            print("Generator optimizer loaded")
            self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
            print("Discriminator optimizer loaded")
            if self.gen_scheduler is not None:
                self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
                print("Generator scheduler loaded")
            if self.disc_scheduler is not None:
                self.disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])
                print("Discriminator scheduler loaded")
            return checkpoint['history']
        else:
            print("No checkpoint found at {}".format(self.path))
            print("Starting from scratch")
            return None

    def save(self, history):
        """
        Saves the generator, discriminator, optimizers and schedulers states and history to the checkpoint file.
        :param history: a dictionary containing the training and validation metrics
        :return:
        """
        print("Saving checkpoint to {}".format(self.path))
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict(),
            'gen_scheduler': self.gen_scheduler.state_dict() if self.gen_scheduler is not None else None,
            'disc_scheduler': self.disc_scheduler.state_dict() if self.disc_scheduler is not None else None,
            'history': history
        }, self.path)
        print("Checkpoint saved")

    def delete(self):
        """
        Deletes the checkpoint file.
        :return:
        """
        if os.path.exists(self.path):
            os.remove(self.path)
            print("Checkpoint deleted")


class Trainer:
    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer, gen_scheduler,
                 disc_scheduler, gen_criterion, disc_criterion, device, train_loader, val_loader,
                 metrics, options):
        """
        :param generator: Generator model
        :param discriminator: Discriminator model
        :param gen_optimizer: Generator optimizer
        :param disc_optimizer: Discriminator optimizer
        :param gen_scheduler: Generator lr scheduler
        :param disc_scheduler: Discriminator lr scheduler
        :param gen_criterion: Generator loss function
        :param disc_criterion: Discriminator loss function
        :param device: Device to use for training
        :param train_loader: Training data loader
        :param val_loader: Validation data loader
        :param metrics: list of instances of metrics objects from torchmetrics
        :param options: Training options, a dictionary
        """

        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.gen_criterion = gen_criterion
        self.disc_criterion = disc_criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics
        self.options = options

        if self.options['clip_weights']:
            self.clipper = ClipWeights(clip_value=self.options['clip_value'])
        else:
            self.clipper = None

        if options['checkpoint_path'] is not None:
            self.checkpoint = Checkpoint(generator, discriminator, gen_optimizer, disc_optimizer,
                                         gen_scheduler, disc_scheduler, options['checkpoint_path'])
            if options['reset_training']:
                self.checkpoint.delete()
            self.history = self.checkpoint.load()

        else:
            self.checkpoint = None
            self.history = None

    def _update_metrics(self, inputs, targets, outputs):
        """
        Updates the metrics for the current batch.
        :param inputs: input images
        :param targets: target images
        :param outputs: generated images
        :return: uiqi value (needs to be treated separately for memory issues)
        """
        if self.options['use_lab_colorspace']:
            # map values in [0, 1] and combine channels, inputs, targets and outputs are in range [-1, 1]
            L = inputs
            L = (L + 1) * 0.5
            ab_true = targets * 0.5 + 0.5
            ab_pred = outputs * 0.5 + 0.5
            true_imgs = torch.cat((L, ab_true), dim=1)
            fake_imgs = torch.cat((L, ab_pred), dim=1)
        else:
            # map to [0, 1] range
            true_imgs = targets
            true_imgs = true_imgs * 0.5 + 0.5
            fake_imgs = outputs
            fake_imgs = fake_imgs * 0.5 + 0.5

        # compute the metrics
        uiqi = 0
        for metric in self.metrics:
            if metric._get_name() == 'FrechetInceptionDistance':
                metric.update(true_imgs, real=True)
                metric.update(fake_imgs, real=False)
            elif metric._get_name() == 'UniversalImageQualityIndex':
                # need to use functional interface
                uiqi = torchmetrics.functional.universal_image_quality_index(fake_imgs, true_imgs,
                                                                             data_range=1.0,
                                                                             reduction=None)
                uiqi = uiqi.mean(axis=(1, 2, 3))
                uiqi = uiqi[~torch.isnan(uiqi)].mean()
            else:
                metric.update(fake_imgs, true_imgs)

        return uiqi

    def _compute_metrics(self, split):
        """
        Computes the metrics for the current epoch.
        :param split: 'train' or 'val'
        :return:
        """
        for metric in self.metrics:
            if metric._get_name() == 'UniversalImageQualityIndex':
                continue
            name = metric._get_name().lower()
            key = '{}_{}'.format(split, name)
            val = metric.compute()
            self.history.setdefault(key, []).append(val.item())
            metric.reset()

    def _plot_images(self, inputs, targets, outputs, split, epoch, step):
        """
        Plots the generated images, side by side with the input and target images.
        :param inputs: input images
        :param targets: target images
        :param outputs: generated images
        :param split: 'train' or 'val'
        :param epoch: current epoch
        :param step: current step
        :return:
        """
        title = '{} images at epoch {} step {}'.format(split, epoch, step)
        # limit the number of images to plot
        n_plot = min(inputs.size(0), 8)
        inputs = inputs[:n_plot]
        targets = targets[:n_plot]
        outputs = outputs[:n_plot]

        if self.options['use_lab_colorspace']:
            # reconstruct the images in RGB color space
            L = inputs.permute(0, 2, 3, 1)
            L = (L + 1) * 50
            ab_true = targets.permute(0, 2, 3, 1)
            ab_true = ab_true * 110
            ab_pred = outputs.permute(0, 2, 3, 1)
            ab_pred = ab_pred * 110
            input_imgs = L.cpu().detach().numpy()
            true_imgs = torch.cat((L, ab_true), dim=3).cpu().detach().numpy()
            fake_imgs = torch.cat((L, ab_pred), dim=3).cpu().detach().numpy()
            true_imgs = lab2rgb(true_imgs)
            fake_imgs = lab2rgb(fake_imgs)

        else:
            input_imgs = inputs.cpu().permute(0, 2, 3, 1).detach().numpy()
            true_imgs = targets.cpu().permute(0, 2, 3, 1).detach().numpy()
            fake_imgs = outputs.cpu().permute(0, 2, 3, 1).detach().numpy()
            # remap the generated image to the range [0, 1] (tanh activation in last layer)
            input_imgs = input_imgs * 0.5 + 0.5
            true_imgs = true_imgs * 0.5 + 0.5
            fake_imgs = fake_imgs * 0.5 + 0.5

        # Plot the first 8 input images, target images and generated images
        fig = plt.figure(figsize=(2 * n_plot, 9))
        for i in range(n_plot):
            ax = fig.add_subplot(3, n_plot, i + 1, xticks=[], yticks=[])
            ax.imshow(input_imgs[i], cmap='gray')
            ax = fig.add_subplot(3, n_plot, i + n_plot + 1, xticks=[], yticks=[])
            ax.imshow(fake_imgs[i])
            ax = fig.add_subplot(3, n_plot, i + 2 * n_plot + 1, xticks=[], yticks=[])
            ax.imshow(true_imgs[i])
        # set title
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        # save figure
        fig.savefig(
            os.path.join(self.options['output_path'],
                         '{}_epoch_{}_{}.png'.format(split, epoch, step)))
        plt.show()

    @staticmethod
    def _set_requires_grad(nets, requires_grad=False):
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        :param nets: a list of networks
        :param requires_grad: whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _train_epoch(self):
        """
        Performs a single training epoch
        :return:
        """
        train_gen_loss = 0
        train_gen_loss_gan = 0
        train_gen_loss_recon = 0
        train_disc_loss = 0
        gp_loss = 0
        uiqi = []  # universal image quality index
        train_iter = tqdm(self.train_loader)

        # set models to train mode
        self.generator.train()
        self.discriminator.train()

        for i, (input_imgs, target_imgs) in enumerate(train_iter):
            # move images to device
            input_imgs = input_imgs.to(self.device, non_blocking=True)
            target_imgs = target_imgs.to(self.device, non_blocking=True)

            # generate fake images (no gradient, do not backpropagate in generator)
            with torch.no_grad():
                fake_images = self.generator(input_imgs)

            # train discriminator
            self._set_requires_grad(self.discriminator, True)
            self.disc_optimizer.zero_grad()
            # combine input images and fake images
            fake_images_to_disc = torch.cat([input_imgs, fake_images], dim=1)
            # get discriminator predictions for fake images
            fake_preds = self.discriminator(fake_images_to_disc)
            # combine input images and real images
            real_images_to_disc = torch.cat([input_imgs, target_imgs], dim=1)
            # get discriminator predictions for real images
            real_preds = self.discriminator(real_images_to_disc)
            # calculate discriminator loss
            disc_loss, gp = self.disc_criterion(input_imgs, real_preds, fake_preds, target_imgs,
                                                fake_images, self.discriminator)
            # backpropagate discriminator loss
            disc_loss.backward()
            # backpropagate gradient penalty
            if gp is not None:
                gp.backward(retain_graph=True)
                disc_loss += gp
                gp_loss += gp.detach().cpu()
            # update discriminator weights
            self.disc_optimizer.step()
            # clip discriminator weights
            if self.clipper is not None:
                self.discriminator.apply(self.clipper)

            train_disc_loss += disc_loss.detach().cpu()

            # train generator every n_critic iterations
            if (i + 1) % self.options['n_critic'] == 0:
                # set requires_grad to False for discriminator (to avoid unnecessary computations)
                self._set_requires_grad([self.discriminator], False)
                self.gen_optimizer.zero_grad()
                # generate fake images
                fake_images = self.generator(input_imgs)
                # combine input images and fake images
                fake_images_to_disc = torch.cat([input_imgs, fake_images], dim=1)
                # get discriminator predictions for fake images
                fake_preds = self.discriminator(fake_images_to_disc)
                # calculate generator loss, gan loss + reconstruction loss (L1 or L2)
                gen_loss_gan, gen_loss_recon = self.gen_criterion(fake_preds, target_imgs,
                                                                  fake_images)
                gen_loss = gen_loss_gan + gen_loss_recon
                # backpropagate generator loss
                gen_loss.backward()
                # update generator weights
                self.gen_optimizer.step()

                # update metric on batch (treat uiqi separately)
                uiqi_ = self._update_metrics(input_imgs.detach(), target_imgs.detach(),
                                             fake_images.detach())
                uiqi.append(uiqi_.detach().cpu().item())

                if gp is not None:
                    display_string = "Gen loss: {:.4f}, Gen gan loss {:.4f}, Gen recon loss {:.4f}, Disc loss: {:.4f}, Disc gan loss {:.4f}, GP :{:.4f}".format(
                        gen_loss, gen_loss_gan, gen_loss_recon, disc_loss, disc_loss - gp, gp)
                else:
                    display_string = "Gen loss: {:.4f}, Gen gan loss {:.4f}, Gen recon loss {:.4f}, Disc loss: {:.4f}".format(
                        gen_loss, gen_loss_gan, gen_loss_recon, disc_loss)
                train_iter.set_description(display_string)
                train_gen_loss += gen_loss.detach().cpu()
                train_gen_loss_gan += gen_loss_gan.detach().cpu()
                train_gen_loss_recon += gen_loss_recon.detach().cpu()

            if (i + 1) % self.options['plot_every_nstep'] == 0 or i == len(self.train_loader) - 1:
                step = i + 1
                if i == len(self.train_loader) - 1:
                    step = 'end'
                self._plot_images(input_imgs, target_imgs, fake_images,
                                  split='train', epoch=self.history['elapsed_epochs'] + 1,
                                  step=step)

        train_gen_loss /= (len(self.train_loader) // self.options['n_critic'])
        train_gen_loss_gan /= (len(self.train_loader) // self.options['n_critic'])
        train_gen_loss_recon /= (len(self.train_loader) // self.options['n_critic'])
        train_disc_loss /= len(self.train_loader)
        gp_loss /= len(self.train_loader)
        uiqi = np.array(uiqi)
        uiqi = uiqi[np.isfinite(uiqi)].mean()
        self.history['train_gen_loss'].append(train_gen_loss.item())
        self.history['train_gen_loss_gan'].append(train_gen_loss_gan.item())
        self.history['train_gen_loss_recon'].append(train_gen_loss_recon.item())
        self.history['train_disc_loss'].append(train_disc_loss.item())
        if isinstance(gp_loss, torch.Tensor):
            self.history.setdefault('train_gp_loss', []).append(gp_loss.item())
        self.history.setdefault('train_universalimagequalityindex', []).append(uiqi.item())

    def _validate_epoch(self):
        """
        Performs a single validation epoch
        :return:
        """
        val_gen_loss = 0
        val_gen_loss_gan = 0
        val_gen_loss_recon = 0
        val_disc_loss = 0
        gp_loss = 0
        uiqi = []  # universal image quality index
        val_iter = tqdm(self.val_loader)

        # TODO: in the paper they use train mode also for validation
        #  (Dropout and BatchNorm in place of noise)
        self.generator.train()
        self.discriminator.train()

        for i, (input_imgs, target_imgs) in enumerate(val_iter):
            # move images to device
            input_imgs = input_imgs.to(self.device, non_blocking=True)
            target_imgs = target_imgs.to(self.device, non_blocking=True)

            with torch.no_grad():
                # generate fake images
                fake_images = self.generator(input_imgs)

                # combine input images and fake images
                fake_images_to_disc = torch.cat([input_imgs, fake_images], dim=1)
                # get discriminator predictions for fake images
                fake_preds = self.discriminator(fake_images_to_disc)
                # combine input images and real images
                real_images_to_disc = torch.cat([input_imgs, target_imgs], dim=1)
                # get discriminator predictions for real images
                real_preds = self.discriminator(real_images_to_disc)

            # calculate discriminator loss
            disc_loss, gp = self.disc_criterion(input_imgs, real_preds, fake_preds, target_imgs,
                                                fake_images, self.discriminator)
            if gp is not None:
                disc_loss += gp
                gp_loss += gp.detach().cpu()

            val_disc_loss += disc_loss.detach().cpu()

            # calculate generator loss, gan loss + reconstruction loss (L1 or L2)
            gen_loss_gan, gen_loss_recon = self.gen_criterion(fake_preds, target_imgs, fake_images)
            gen_loss = gen_loss_gan + gen_loss_recon

            # update metric on batch (treat uiqi separately)
            uiqi_ = self._update_metrics(input_imgs.detach(), target_imgs.detach(),
                                         fake_images.detach())
            uiqi.append(uiqi_.detach().cpu().item())

            if gp is not None:
                display_string = "Gen loss: {:.4f}, Gen gan loss {:.4f}, Gen recon loss {:.4f}, Disc loss: {:.4f}, Disc gan loss {:.4f}, GP :{:.4f}".format(
                    gen_loss, gen_loss_gan, gen_loss_recon, disc_loss, disc_loss - gp, gp)
            else:
                display_string = "Gen loss: {:.4f}, Gen gan loss {:.4f}, Gen recon loss {:.4f}, Disc loss: {:.4f}".format(
                    gen_loss, gen_loss_gan, gen_loss_recon, disc_loss)

            val_iter.set_description(display_string)
            val_gen_loss += gen_loss.detach().cpu()
            val_gen_loss_gan += gen_loss_gan.detach().cpu()
            val_gen_loss_recon += gen_loss_recon.detach().cpu()

            if i == len(self.val_loader) - 1:
                self._plot_images(input_imgs, target_imgs, fake_images,
                                  split='val', epoch=self.history['elapsed_epochs'] + 1, step='end')

        val_gen_loss /= len(self.val_loader)
        val_gen_loss_gan /= len(self.val_loader)
        val_gen_loss_recon /= len(self.val_loader)
        val_disc_loss /= len(self.val_loader)
        gp_loss /= len(self.val_loader)
        uiqi = np.array(uiqi)
        uiqi = uiqi[np.isfinite(uiqi)].mean()
        self.history['val_gen_loss'].append(val_gen_loss.item())
        self.history['val_gen_loss_gan'].append(val_gen_loss_gan.item())
        self.history['val_gen_loss_recon'].append(val_gen_loss_recon.item())
        self.history['val_disc_loss'].append(val_disc_loss.item())
        if isinstance(gp_loss, torch.Tensor):
            self.history.setdefault('val_gp_loss', []).append(gp_loss.item())
        self.history.setdefault('val_universalimagequalityindex', []).append(uiqi.item())

    def train(self):
        """
        Trains the generator and discriminator models
        :return: history, a dictionary containing the training and validation metrics
        """
        # TODO: add other metrics
        if self.history is None:
            self.history = {'train_gen_loss': [], 'train_gen_loss_gan': [],
                            'train_gen_loss_recon': [], 'train_disc_loss': [],
                            'val_gen_loss': [], 'val_gen_loss_gan': [],
                            'val_gen_loss_recon': [], 'val_disc_loss': [],
                            'elapsed_epochs': 0, 'epoch_times': []}

        current_epoch = self.history['elapsed_epochs']
        while current_epoch < self.options['num_epochs']:
            start_time = time.time()
            print("\n\n")
            print("Epoch {}/{}".format(current_epoch + 1, self.options['num_epochs']))

            # train
            print("Training...")
            self._train_epoch()
            # compute epoch metrics
            self._compute_metrics('train')

            # validate
            print("Validation...")
            self._validate_epoch()
            # compute epoch metrics
            self._compute_metrics('val')

            # update lr schedulers
            if self.disc_scheduler is not None:
                self.disc_scheduler.step()
            if self.gen_scheduler is not None:
                self.gen_scheduler.step()

            end_time = time.time()
            self.history['epoch_times'].append(end_time - start_time)

            # print epoch summary
            print("Epoch {} took {}".format(
                current_epoch + 1, format_time(self.history['epoch_times'][-1])))
            print("Train Generator Loss: {:.4f}".format(
                self.history['train_gen_loss'][-1]))
            print("Train Generator Loss GAN: {:.4f}".format(
                self.history['train_gen_loss_gan'][-1]))
            print("Train Generator Loss Recon: {:.4f}".format(
                self.history['train_gen_loss_recon'][-1]))
            print("Train Discriminator Loss: {:.4f}".format(
                self.history['train_disc_loss'][-1]))
            if 'train_gp_loss' in self.history:
                print("Train Gradient Penalty Loss: {:.4f}".format(
                    self.history['train_gp_loss'][-1]))
            print('\n')
            print("Validation Generator Loss: {:.4f}".format(
                self.history['val_gen_loss'][-1]))
            print("Validation Generator Loss GAN: {:.4f}".format(
                self.history['val_gen_loss_gan'][-1]))
            print("Validation Generator Loss Recon: {:.4f}".format(
                self.history['val_gen_loss_recon'][-1]))
            print("Validation Discriminator Loss: {:.4f}".format(
                self.history['val_disc_loss'][-1]))
            if 'val_gp_loss' in self.history:
                print("Validation Gradient Penalty Loss: {:.4f}".format(
                    self.history['val_gp_loss'][-1]))
            print('\n')
            for metric in self.metrics:
                name = metric._get_name()
                print("Train {}: {:.4f}".format(
                    name, self.history['train_' + name.lower()][-1]))
                print("Validation {}: {:.4f}".format(
                    name, self.history['val_' + name.lower()][-1]))

            # clear output every 10 epochs
            if (current_epoch + 1) % 10 == 0:
                clear_output(wait=True)

            # update epoch counter
            current_epoch += 1
            self.history['elapsed_epochs'] = current_epoch

            # save checkpoint
            if self.checkpoint is not None:
                self.checkpoint.save(self.history)

        return self.history


class ClipWeights(object):
    def __init__(self, clip_value=0.01):
        self.clip_value = clip_value

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.clip_value, self.clip_value)
            module.weight.data = w
