import datetime
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def format_time(elapsed):
    """
    :param elapsed: time in seconds
    :return: formatted time string in hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def generate_and_save_images(generator, epoch, gen_input):
    """
    Generates and saves rgb images
    :param generator: generator model
    :param epoch: current epoch
    :param gen_input: noise input for generator
    :return:
    """
    current_mode = generator.training

    generator.eval()
    predictions = generator(gen_input)
    # TODO: check if values need to be scaled
    grid = make_grid(predictions, 4).numpy().squeeze().transpose(1, 2, 0)

    plt.imshow(grid.astype(np.uint8))
    plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

    # reset generator to previous mode
    generator.train() if current_mode else generator.eval()

