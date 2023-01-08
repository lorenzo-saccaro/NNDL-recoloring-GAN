import torch


def discriminator_loss(real_output, fake_output, device):
    """
    Calculates discriminator loss
    :param real_output: real image output of discriminator
    :param fake_output: fake image output of discriminator
    :param device: device to run on
    :return: discriminator loss
    """
    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        real_output, torch.ones_like(real_output, device=device))
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        fake_output, torch.zeros_like(fake_output, device=device))
    return real_loss + fake_loss


def generator_loss(fake_output, device):
    """
    Calculates generator loss
    :param fake_output: fake image output of discriminator
    :param device: device to run on
    :return: generator loss
    """
    return torch.nn.functional.binary_cross_entropy_with_logits(
        fake_output, torch.ones_like(fake_output, device=device))


def L1_loss(real_image, fake_image):
    """
    Calculates L1 loss
    :param real_image: real image
    :param fake_image: fake image from generator
    :return: L1 loss
    """
    return torch.nn.functional.l1_loss(real_image, fake_image)


def L2_loss(real_image, fake_image):
    """
    Calculates L2 loss
    :param real_image: real image
    :param fake_image: fake image from generator
    :return: L2 loss
    """
    return torch.nn.functional.mse_loss(real_image, fake_image)


class DiscriminatorCriterion:

    def __init__(self, device):
        """
        Discriminator loss
        :param device: device to run on
        """
        self.device = device

    def __call__(self, real_output, fake_output):
        """
        Calculates discriminator loss
        :param real_output: real image output of discriminator
        :param fake_output: fake image output of discriminator
        :return:
        """
        return discriminator_loss(real_output, fake_output, self.device)


class GeneratorCriterion:

    def __init__(self, device, use_l1_loss=False, use_l2_loss=False, l1_lambda=0, l2_lambda=0):
        """
        Generator loss
        :param device: device to run on
        :param use_l1_loss: use L1 loss
        :param use_l2_loss: use L2 loss
        :param l1_lambda: L1 loss lambda
        :param l2_lambda: L2 loss lambda
        """
        self.device = device
        self.use_l1_loss = use_l1_loss
        self.use_l2_loss = use_l2_loss
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def __call__(self, fake_output, real_image, fake_image):
        """
        :param fake_output: fake image output of discriminator
        :param real_image: real image
        :param fake_image: fake image from generator
        :return: generator loss
        """
        loss = generator_loss(fake_output, self.device)
        if self.use_l1_loss:
            loss += self.l1_lambda * L1_loss(real_image, fake_image)
        if self.use_l2_loss:
            loss += self.l2_lambda * L2_loss(real_image, fake_image)
        return loss
