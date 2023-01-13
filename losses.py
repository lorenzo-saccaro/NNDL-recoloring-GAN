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


def Wasserstein_loss(real_output, fake_output):
    """
    Calculates Wasserstein loss, note that we are using +1 as the label for fakes and -1 as the label for reals
    (pass real_output =0 when computing loss for generator)
    :param real_output: real image output of discriminator
    :param fake_output: fake image output of discriminator
    :return: Wasserstein loss
    """
    return torch.mean(fake_output) - torch.mean(real_output)


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

    def __init__(self, device, wgan=False, wgan_gp=False, gp_lambda=10, gp_type='mixed',
                 gp_constant=1):
        """
        Discriminator loss
        :param device: device to run on
        :param wgan: whether to use Wasserstein loss
        :param wgan_gp: whether to use gradient penalty
        :param gp_lambda: gradient penalty lambda
        :param gp_type: gradient penalty type, can be 'mixed', 'real', 'fake'
        :param gp_constant: gradient penalty constant
        """
        self.device = device
        self.wgan = wgan
        self.wgan_gp = wgan_gp
        assert (wgan if wgan_gp else True), 'WGAN-GP must be used with WGAN'
        self.gp_lambda = gp_lambda
        # real uses real image, fake uses fake image, mixed uses both
        assert gp_type in ['mixed', 'real', 'fake'], 'Invalid gradient penalty type: ' + gp_type
        self.gp_type = gp_type
        self.gp_constant = gp_constant

    def __call__(self, real_output, fake_output, real_images=None, fake_images=None,
                 discriminator=None):
        """
        Calculates discriminator loss
        :param real_output: real image output of discriminator
        :param fake_output: fake image output of discriminator
        :param real_images: real images
        :param fake_images: fake images
        :param discriminator: discriminator instance
        :return:
        """
        if self.wgan:
            loss = Wasserstein_loss(real_output, fake_output)
            if self.wgan_gp:
                loss += self._gradient_penalty(real_images, fake_images, discriminator)
            return loss
        else:
            return discriminator_loss(real_output, fake_output, self.device)

    def _gradient_penalty(self, real_images, fake_images, discriminator):
        """
        Calculates gradient penalty
        :param real_images: real images
        :param fake_images: fake images
        :param discriminator: discriminator
        :return: gradient penalty
        """
        if self.gp_type == 'real':
            interpolates = real_images
        elif self.gp_type == 'fake':
            interpolates = fake_images
        elif self.gp_type == 'mixed':
            # Random weight term for interpolation between real and fake data
            alpha = torch.randn((real_images.size(0), 1, 1, 1), device=self.device)
            # Get random interpolation between real and fake data
            interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
        else:
            raise ValueError("Invalid gradient penalty type")

        disc_interpolates = discriminator(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size(), device=self.device, requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=grad_outputs, create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = torch.mean(
            ((gradients + 1e-16).norm(2, dim=1) - self.gp_constant) ** 2)

        return gradient_penalty * self.gp_lambda


class GeneratorCriterion:

    def __init__(self, device, wgan=False, use_l1_loss=False, use_l2_loss=False, l1_lambda=0,
                 l2_lambda=0):
        """
        Generator loss
        :param device: device to run on
        :param wgan: whether to use Wasserstein loss
        :param use_l1_loss: use L1 loss
        :param use_l2_loss: use L2 loss
        :param l1_lambda: L1 loss lambda
        :param l2_lambda: L2 loss lambda
        """
        self.device = device
        self.wgan = wgan
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
        if self.wgan:
            loss = Wasserstein_loss(real_output=0, fake_output=fake_output)
        else:
            loss = generator_loss(fake_output, self.device)
        if self.use_l1_loss:
            loss += self.l1_lambda * L1_loss(real_image, fake_image)
        if self.use_l2_loss:
            loss += self.l2_lambda * L2_loss(real_image, fake_image)
        return loss
