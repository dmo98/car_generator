import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
print('Completed importing libraries', '\n')

SAVED_MODEL_DIR = 'trained_models'
SAVED_MODEL = 'DCGAN_v1_trained_for_50_epochs.h5'
LATENT_DIM = 100
SAVE_IMAGES_TO_DIR = 'generated_images'
# Load the model
dcgan = keras.models.load_model(os.path.join(SAVED_MODEL_DIR, SAVED_MODEL))
print('Loaded the model', '\n')


def generate_images(dcgan_model, num_imgs, save_to_file, save_image=True):
    """
    Samples 'num_imgs' random points from a Gaussian distribution, passes them to the Generator of the DCGAN('dcgan_model')
    to generate 'num_imgs' corresponding images, and plots them (5 per row)
    """
    generator, discriminator = dcgan_model.layers

    random_latent_vectors = tf.random.normal(shape=[num_imgs, LATENT_DIM])
    generated_images = generator(random_latent_vectors)
    generated_images.numpy()

    # predicted_labels = discriminator(generated_images)

    columns = 5
    rows = num_imgs // columns
    plt.figure(figsize=(columns * 3, rows * 3))
    for row in range(rows):
        for col in range(columns):
            index = columns * row + col
            plt.subplot(rows, columns, index + 1)
            plt.imshow(((generated_images[index] + 1) / 2)[:, :, ::-1], interpolation="nearest")
            # plt.title('P(real) = {}'.format(predicted_labels[index]))
            plt.axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    if save_image:
        plt.savefig(save_to_file)
    plt.show()


if os.path.isdir(SAVE_IMAGES_TO_DIR):  # If this directory already exists, save the generated images in it.
    saved_images_file = os.path.join(SAVE_IMAGES_TO_DIR, 'Images_of_Cars_Generated_using_{}.jpg'.format(SAVED_MODEL))
    generate_images(dcgan, 15, save_to_file=saved_images_file)
else:                                  # Otherwise, make that directory and then save the generated images in it.
    os.mkdir(SAVE_IMAGES_TO_DIR)
    saved_images_file = os.path.join(SAVE_IMAGES_TO_DIR, 'Images_of_Cars_Generated_using_{}.jpg'.format(SAVED_MODEL))
    generate_images(dcgan, 15, save_to_file=saved_images_file)