import os
import time
from utils import format_time
from tqdm import tqdm
import torch


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

        if options['checkpoint_path'] is not None:
            self.checkpoint = Checkpoint(generator, discriminator, gen_optimizer, disc_optimizer,
                                         gen_scheduler, disc_scheduler, options['checkpoint_path'])
            if options['reset_training']:
                self.checkpoint.delete()
            self.history = self.checkpoint.load()

        else:
            self.checkpoint = None
            self.history = None

    def _train_epoch(self):
        """
        Performs a single training epoch
        :return:
        """
        train_gen_loss = 0
        train_disc_loss = 0
        train_iter = tqdm(self.train_loader)

        for batch in train_iter:
            batch = batch.to(self.device)

            # set models to train mode
            self.generator.train()
            self.discriminator.train()

            # get images from batch
            input_imgs, target_imgs = batch

            # generate fake images
            fake_images = self.generator(input_imgs)

            # train discriminator

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
            disc_loss = self.disc_criterion(fake_preds, real_preds, target_imgs, fake_images,
                                            self.discriminator)
            # backpropagate discriminator loss
            disc_loss.backward()
            # update discriminator weights
            self.disc_optimizer.step()

            # train generator

            self.gen_optimizer.zero_grad()
            # combine input images and fake images
            fake_images_to_disc = torch.cat([input_imgs, fake_images], dim=1)
            # get discriminator predictions for fake images
            fake_preds = self.discriminator(fake_images_to_disc)
            # calculate generator loss
            gen_loss = self.gen_criterion(fake_preds, target_imgs, fake_images, self.device)
            # backpropagate generator loss
            gen_loss.backward()
            # update generator weights
            self.gen_optimizer.step()

            train_iter.set_description(
                "Gen loss: {:.4f}, Disc loss: {:.4f}".format(gen_loss, disc_loss))
            train_gen_loss += gen_loss.detach().cpu()
            train_disc_loss += disc_loss.detach().cpu()

        train_gen_loss /= len(self.train_loader)
        train_disc_loss /= len(self.train_loader)
        self.history['train_gen_loss'].append(train_gen_loss)
        self.history['train_disc_loss'].append(train_disc_loss)

    def _validate_epoch(self):
        """
        Performs a single validation epoch
        :return:
        """
        val_gen_loss = 0
        val_disc_loss = 0
        val_iter = tqdm(self.val_loader)

        for batch in val_iter:
            batch = batch.to(self.device)

            # TODO: in the paper they use train mode also for validation
            #  (Dropout and BatchNorm in place of noise)
            self.generator.eval()
            self.discriminator.eval()

            # get images from batch
            input_imgs, target_imgs = batch

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
                disc_loss = self.disc_criterion(fake_preds, real_preds, target_imgs, fake_images,
                                                self.discriminator)
                # calculate generator loss
                gen_loss = self.gen_criterion(fake_preds, target_imgs, fake_images)

            val_iter.set_description(
                "Gen loss: {:.4f}, Disc loss: {:.4f}".format(gen_loss, disc_loss))
            val_gen_loss += gen_loss.detach().cpu()
            val_disc_loss += disc_loss.detach().cpu()

        val_gen_loss /= len(self.val_loader)
        val_disc_loss /= len(self.val_loader)
        self.history['val_gen_loss'].append(val_gen_loss)
        self.history['val_disc_loss'].append(val_disc_loss)

    def train(self):
        """
        Trains the generator and discriminator models
        :return: history, a dictionary containing the training and validation metrics
        """
        # TODO: add other metrics
        if self.history is None:
            self.history = {'train_gen_loss': [], 'train_disc_loss': [],
                            'val_gen_loss': [], 'val_disc_loss': [],
                            'elapsed_epochs': 0}

        current_epoch = self.history['elapsed_epochs']
        while current_epoch < self.options['num_epochs']:
            start_time = time.time()
            print("\n\n")
            print("Epoch {}/{}".format(current_epoch + 1, self.options['epochs']))

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

            # print epoch summary
            print("Epoch {} took {}".format(current_epoch + 1, format_time(end_time - start_time)))
            print("Train Generator Loss: {:.4f}".format(self.history['train_gen_loss'][-1]))
            print("Train Discriminator Loss: {:.4f}".format(self.history['train_disc_loss'][-1]))
            print("Validation Generator Loss: {:.4f}".format(self.history['val_gen_loss'][-1]))
            print("Validation Discriminator Loss: {:.4f}".format(self.history['val_disc_loss'][-1]))

            # update epoch counter
            current_epoch += 1
            self.history['elapsed_epochs'] = current_epoch

            # save checkpoint
            if self.checkpoint is not None:
                self.checkpoint.save(self.history)

        return self.history


class WTrainer(Trainer):

    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer, gen_scheduler,
                 disc_scheduler, gen_criterion, disc_criterion, device, train_loader, val_loader,
                 options):
        super().__init__(generator, discriminator, gen_optimizer, disc_optimizer, gen_scheduler,
                         disc_scheduler, gen_criterion, disc_criterion, device, train_loader,
                         val_loader, options)

        if self.options['clip_weights']:
            self.clipper = ClipWeights()
        else:
            self.clipper = None

    def _train_epoch(self):
        """
        Performs a single training epoch
        :return:
        """
        train_gen_loss = 0
        train_disc_loss = 0
        train_iter = tqdm(self.train_loader)

        for i, batch in enumerate(train_iter):
            batch = batch.to(self.device)

            # set models to train mode
            self.generator.train()
            self.discriminator.train()

            # get images from batch
            input_imgs, target_imgs = batch

            # generate fake images
            fake_images = self.generator(input_imgs)

            # train discriminator

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
            disc_loss = self.disc_criterion(fake_preds, real_preds, target_imgs, fake_images,
                                            self.discriminator)
            # backpropagate discriminator loss
            disc_loss.backward()
            # update discriminator weights
            self.disc_optimizer.step()
            # clip discriminator weights
            if self.clipper is not None:
                self.discriminator.apply(self.clipper)

            train_disc_loss += disc_loss.detach().cpu()

            # train generator
            if (i + 1) % self.options['n_critic'] == 0:
                self.gen_optimizer.zero_grad()
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

        train_gen_loss /= (len(self.train_loader) // self.options['n_critic'])
        train_disc_loss /= len(self.train_loader)
        self.history['train_gen_loss'].append(train_gen_loss)
        self.history['train_disc_loss'].append(train_disc_loss)


class ClipWeights(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            print("Entered")
            w = module.weight.data
            w = w.clamp(-1, 1)
            module.weight.data = w
