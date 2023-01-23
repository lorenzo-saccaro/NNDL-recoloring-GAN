import os
import time
from utils import format_time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
from skimage.color import lab2rgb


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
                 options):
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
            fake_imgs = fake_imgs * 0.5 + 0.5

        # Plot the first 8 input images, target images and generated images
        fig = plt.figure(figsize=(2*n_plot, 9))
        for i in range(n_plot):
            ax = fig.add_subplot(3, n_plot, i + 1, xticks=[], yticks=[])
            ax.imshow(input_imgs[i], cmap='gray')
            ax = fig.add_subplot(3, n_plot, i + n_plot + 1, xticks=[], yticks=[])
            ax.imshow(fake_imgs[i])
            ax = fig.add_subplot(3, n_plot, i + 2*n_plot + 1, xticks=[], yticks=[])
            ax.imshow(true_imgs[i])
        # set title
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        # save figure
        fig.savefig(
            os.path.join(self.options['output_path'], '{}_epoch_{}_{}.png'.format(split, epoch, step)))
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
        train_disc_loss = 0
        train_iter = tqdm(self.train_loader)

        # set models to train mode
        self.generator.train()
        self.discriminator.train()

        for i, (input_imgs, target_imgs) in enumerate(train_iter):
            # move images to device
            input_imgs = input_imgs.to(self.device, non_blocking=True)
            target_imgs = target_imgs.to(self.device, non_blocking=True)

            # generate fake images
            fake_images = self.generator(input_imgs)

            # train discriminator
            self._set_requires_grad(self.discriminator, True)
            self.disc_optimizer.zero_grad()
            # combine input images and fake images
            fake_images_to_disc = torch.cat([input_imgs, fake_images], dim=1)
            # get discriminator predictions for fake images
            fake_preds = self.discriminator(fake_images_to_disc.detach())
            # combine input images and real images
            real_images_to_disc = torch.cat([input_imgs, target_imgs], dim=1)
            # get discriminator predictions for real images
            real_preds = self.discriminator(real_images_to_disc)
            # calculate discriminator loss
            disc_loss = self.disc_criterion(input_imgs, real_preds, fake_preds, target_imgs,
                                            fake_images, self.discriminator)
            # backpropagate discriminator loss
            disc_loss.backward()
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
                # calculate generator loss
                gen_loss = self.gen_criterion(fake_preds, target_imgs, fake_images)
                # backpropagate generator loss
                gen_loss.backward()
                # update generator weights
                self.gen_optimizer.step()

                train_iter.set_description(
                    "Gen loss: {:.4f}, Disc loss: {:.4f}".format(gen_loss, disc_loss))
                train_gen_loss += gen_loss.detach().cpu()
                train_disc_loss += disc_loss.detach().cpu()

            if (i + 1) % self.options['plot_every_nstep'] == 0 or i == len(self.train_loader) - 1:
                step = i + 1
                if i == len(self.train_loader) - 1:
                    step = 'end'
                self._plot_images(input_imgs, target_imgs, fake_images,
                                  split='train', epoch=self.history['elapsed_epochs']+1, step=step)

        train_gen_loss /= (len(self.train_loader) // self.options['n_critic'])
        train_disc_loss /= len(self.train_loader)
        self.history['train_gen_loss'].append(train_gen_loss.item())
        self.history['train_disc_loss'].append(train_disc_loss.item())

    def _validate_epoch(self):
        """
        Performs a single validation epoch
        :return:
        """
        val_gen_loss = 0
        val_disc_loss = 0
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
            disc_loss = self.disc_criterion(input_imgs, real_preds, fake_preds, target_imgs,
                                            fake_images, self.discriminator)
            # calculate generator loss
            gen_loss = self.gen_criterion(fake_preds, target_imgs, fake_images)

            val_iter.set_description(
                "Gen loss: {:.4f}, Disc loss: {:.4f}".format(gen_loss, disc_loss))
            val_gen_loss += gen_loss.detach().cpu()
            val_disc_loss += disc_loss.detach().cpu()

            if i == len(self.val_loader) - 1:
                self._plot_images(input_imgs, target_imgs, fake_images,
                                  split='val', epoch=self.history['elapsed_epochs'] + 1, step='end')

        val_gen_loss /= len(self.val_loader)
        val_disc_loss /= len(self.val_loader)
        self.history['val_gen_loss'].append(val_gen_loss.item())
        self.history['val_disc_loss'].append(val_disc_loss.item())

    def train(self):
        """
        Trains the generator and discriminator models
        :return: history, a dictionary containing the training and validation metrics
        """
        # TODO: add other metrics
        if self.history is None:
            self.history = {'train_gen_loss': [], 'train_disc_loss': [],
                            'val_gen_loss': [], 'val_disc_loss': [],
                            'elapsed_epochs': 0, 'epoch_times': []}

        current_epoch = self.history['elapsed_epochs']
        while current_epoch < self.options['num_epochs']:
            start_time = time.time()
            print("\n\n")
            print("Epoch {}/{}".format(current_epoch + 1, self.options['num_epochs']))

            # train
            print("Training...")
            self._train_epoch()

            # validate
            print("Validation...")
            self._validate_epoch()

            # update lr schedulers
            if self.disc_scheduler is not None:
                self.disc_scheduler.step()
            if self.gen_scheduler is not None:
                self.gen_scheduler.step()

            end_time = time.time()
            self.history['epoch_times'].append(end_time - start_time)

            # print epoch summary
            print("Epoch {} took {}".format(current_epoch + 1, format_time(self.history['epoch_times'][-1])))
            print("Train Generator Loss: {:.4f}".format(self.history['train_gen_loss'][-1]))
            print("Train Discriminator Loss: {:.4f}".format(self.history['train_disc_loss'][-1]))
            print("Validation Generator Loss: {:.4f}".format(self.history['val_gen_loss'][-1]))
            print("Validation Discriminator Loss: {:.4f}".format(self.history['val_disc_loss'][-1]))

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
