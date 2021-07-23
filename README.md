# car_generator

In this project, I have trained a DCGAN on the Stanford Cars Dataset to generate images of cars.

This repository contains the following files:
1.) prepare_data.py: It reads the images, crops them using the bounding boxes provided in the annotations file, resizes them to (64,64,3) and stores all the images in a .npy file.
2.) DCGAN_trained_on_(64,64,3)_images.ipynb: I used Google Colab to train the DCGAN on the images stored in the .npy file
3.) generate_images.py: It uses the saved model stored in the 'trained_models' folder to generate images of cars given latent vectors sampled from a Gaussian distribution.
