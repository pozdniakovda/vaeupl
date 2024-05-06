import os
import sys
import cv2
import gc
from time import sleep
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

# set the width of the terminal to 80
term_columns = 80

torch.backends.cudnn.enabled = False

# decorations
# print("Start...")
# check if CUDA is available and set device to cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")


# print(torch.__version__)
# print("CUDA Available:", torch.cuda.is_available())
# print("CUDA Version:", torch.version.cuda)

# this class loads images from a dataset and pads them into a black square
class ImageLoader:
    def __init__(self, dataset):

        print("-" * term_columns)

        # dataset images exist in folder dataset/hr_train or etc.
        # check if already exists.

        self.dataset_paths = {
            "hr_train": "DIV2K/DIV2K_train_HR",
            "hr_valid": "DIV2K/DIV2K_valid_HR",
            "lr_train": "DIV2K/DIV2K_train_LR_bicubic",
            "lr_valid": "DIV2K/DIV2K_valid_LR_bicubic",
        }
        self.dataset = dataset

        # check if the images have already been processed
        files_are_processed = True
        files = os.listdir(self.dataset_paths[dataset])

        for file in files:
            if not file.endswith("_padded.png"):
                files_are_processed = False
                break

        if not files_are_processed:
            # preparing the corresponding padded images dataset
            print(f"STATUS: Preparing dataset {dataset}")

            self.min_dim = None
            self.max_dim = None

            # load images from dataset
            self.images = self.load_images()

            # add padding and check images for squareness
            self.prepare_images()

            print(f"STATUS: {dataset} prepared!")
        else:
            # get self.max_dim from the first image since all of them are the same size now
            files = os.listdir(self.dataset_paths[self.dataset])
            image = cv2.imread(os.path.join(self.dataset_paths[self.dataset], files[0]))
            self.max_dim = image.shape[0]

            print(f"STATUS: {dataset} already prepared, skipping...")

        print("-" * term_columns)

    def __len__(self):
        return len(self.images)

    def image_to_tensor(self, index):

        # print("STATUS: Processing a padded image.")

        # check index validity and reconstruct the filename from index
        filename = None

        match self.dataset:
            case "hr_train":
                if index in range(1, 801):
                    # print("STATUS: Image index valid.")
                    filename = str(index).zfill(4) + "_padded.png"
                else:
                    # print("STATUS: Image index invalid.")
                    return 0
            case "lr_train":
                if index in range(1, 801):
                    # print("STATUS: Image index valid.")
                    filename = str(index).zfill(4) + "x8_padded.png"
                else:
                    # print("STATUS: Image index invalid.")
                    return 0
            case "hr_valid":
                if index in range(801, 901):
                    # print("STATUS: Image index valid.")
                    filename = str(index).zfill(4) + "_padded.png"
                else:
                    # print("STATUS: Image index invalid.")
                    return 0
            case "lr_valid":
                if index in range(801, 901):
                    # print("STATUS: Image index valid.")
                    filename = str(index).zfill(4) + "x8_padded.png"
                else:
                    # print("STATUS: Image index invalid.")
                    return 0
            case _:
                # print("ERROR: no such dataset!")
                return 0

        # print(f"STATUS: Loading image {filename} from dataset {self.dataset} into memory...")

        img_loaded = None

        # check if the files are valid
        if filename.endswith(".png"):
            img_path = os.path.join(self.dataset_paths[self.dataset], filename)

            # load the img from the path using cv2
            img = cv2.imread(img_path)

            # convert from BGR (cv2 standard) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # check if img was loaded successfully
            if img is not None:
                img_loaded = img
            # else:
                # print(f"ERROR: Image at {img_path} could not be loaded.")

        # print(f"STATUS: Image {filename} loaded from dataset {self.dataset}")

        # print(f"STATUS: Converting image {filename} to tensor...")
        # convert image to tensor, permute dimensions to (c, h, w) for pytorch, normalize to range [0,1]
        img_tensor = torch.from_numpy(img_loaded).permute(2, 0, 1).float() / 255.0

        # add a batch dimension to make it into [1, 3, 255, 255]
        img_tensor = img_tensor.unsqueeze(0)
        # print(f"STATUS: Image {filename} converted to tensor!")

        return img_tensor

    # load all png images from a path
    def load_images(self):
        # checking for valid dataset name
        if self.dataset_paths[self.dataset] is None:
            print(f"ERROR: {self.dataset} is not a valid dataset")
            return 0

        print(f"STATUS: Loading images from dataset {self.dataset} into memory...")

        loaded_images = {}
        for filename in tqdm(os.listdir(self.dataset_paths[self.dataset]), ncols=term_columns):
            # to make the progress bar a little easier to look at
            sleep(0.002)

            # check if the files are valid
            if filename.endswith(".png"):
                img_path = os.path.join(self.dataset_paths[self.dataset], filename)

                # load the img from the path using cv2
                img = cv2.imread(img_path)

                # convert from BGR (cv2 standard) to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # check if img was loaded successfully
                if img is not None:
                    loaded_images[img_path] = img
                else:
                    print(f"ERROR: Image at {img_path} could not be loaded.")

        print("STATUS: Images loaded!")
        return loaded_images

    # plot an image from the dataset using its index
    def show_image(self, index):
        # reconstruct the image name from index
        filename = None
        if self.dataset in ["hr_train", "hr_valid"]:
            filename = str(index).zfill(4) + ".png"
        else:
            filename = str(index).zfill(4) + "x8.png"

        # get full image path and get image from images
        img_path = os.path.join(self.dataset_paths[self.dataset], filename)
        img = self.images[img_path]

        # plot an image to show it
        plt.imshow(img)
        plt.title(f'{filename}')
        plt.show()
        print(f"STATUS: Image at {img_path} is plotted.")

    def prepare_images(self):
        # to make sure it is not run on an empty dictionary
        if self.images is None:
            print("ERROR: Images need to be loaded first.")
            return 0

        print("STATUS: Processing images...")

        # calculate the min/max dimension across all images for padding to square shape
        min_dim = sys.maxsize
        max_dim = 0
        for img in self.images.values():
            cur_min = min(img.shape[:2])
            cur_max = max(img.shape[:2])
            if cur_min < min_dim:
                min_dim = cur_min
            if cur_max > max_dim:
                max_dim = cur_max

        # save the min_dim and max_dim values
        self.min_dim = min_dim
        self.max_dim = max_dim

        # pad all images to the max_dim size
        padded_images = {}
        for filename, img in tqdm(self.images.items(), ncols=term_columns):
            sleep(0.005)

            # calculate the amount of padding needed
            height, width = img.shape[:2]
            padding_top = (max_dim - height) // 2
            padding_bottom = max_dim - height - padding_top
            padding_left = (max_dim - width) // 2
            padding_right = max_dim - width - padding_left

            # apply black padding to the images so that they are centered
            padded_img = cv2.copyMakeBorder(img,
                                            top=padding_top,
                                            bottom=padding_bottom,
                                            left=padding_left,
                                            right=padding_right,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=[0, 0, 0])

            # replace original image with padded image
            self.images[filename] = padded_img

            # make sure the directory for the new dataset exists
            os.makedirs(os.path.dirname(filename), exist_ok = True)

            # save image with a new name to indicate it is prepared
            cv2.imwrite(filename=filename[:-4] + "_padded.png", img=padded_img)

            # delete initial filename
            os.remove(filename)

        # check if all images are max_dim by max_dim
        non_matching_img = {}
        for filename, img in self.images.items():
            height, width = img.shape[:2]
            if height != max_dim or width != max_dim:
                non_matching_img[filename] = img
        if not non_matching_img:
            print("STATUS: All images are correct size.")
        else:
            for filename, img in non_matching_img.items():
                print(f"ERROR: Image at {filename} is not a square image.")

        print("STATUS: Images processed!")

        # free some memory
        del self.images
        print("STATUS: Images deleted from memory.")

    def images_to_tensor_list(self, batch_size=100):

        # to make sure it is not run on an empty dictionary
        if self.images is None:
            print("ERROR: Images need to be loaded first.")
            return 0

        print("STATUS: Converting images to tensor list for the encoder...")

        # constructing tensor list from images
        tensor_list = []
        images_list = list(self.images.values())

        for i in tqdm(range(0, len(images_list), batch_size), ncols=80):
            # current batch of tensors
            batch_tensors = []

            # current batch of images
            batch = images_list[i:i + batch_size]

            # Process each image in the batch
            for img in batch:
                # convert image to tensor and permute dimensions to (c, h, w) for pytorch
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)

                # normalize the tensor to range [0, 1]
                img_tensor = img_tensor.float() / 255.0

                # append the processed tensor to the list
                batch_tensors.append(img_tensor)

            # Extend the main tensor_list with the tensors from the current batch
            tensor_list.extend(batch_tensors)

        self.tensor_list = tensor_list

        print("STATUS: Tensor list assembled!")

# testing class ImageLoader
# image_loader = ImageLoader("lr_train")
# image_loader.load_images()
# image_loader.prepare_images()
# image_loader.show_image(1)

# loading hr_train dataset (images 0001 to 0800)
hr_train = ImageLoader("hr_train")

# loading hr_valid dataset (images 0801 to 0900)
hr_valid = ImageLoader("hr_valid")

# loading lr_train dataset (compressed images 0001 to 0800)
lr_train = ImageLoader("lr_train")

# loading lr_valid dataset (compressed images 0801 to 0900)
lr_valid = ImageLoader("lr_valid")

# checking if all the image tensors are correct shape
# for index in range(1,20):
# tensor = lr_train.image_to_tensor(index)
# print(tensor.shape)


# VAE encoder
class VAEEncoder(nn.Module):
    def __init__(self, max_dim, input_channels, latent_dim):
        # call the constructor of nn.Module
        nn.Module.__init__(self)

        # max_dim is 255
        # define four convolutional layers: RGB to 32, 64, 128, 256
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # 128 dim
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 dim
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32 dim
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),  # 16 dim
            nn.ReLU()
        )

        # compute the size of the output from the last convolutional layer to link it to the dense layers correctly
        self.flattened_size = 256 * (max_dim // 16) * (max_dim // 16)
        # print(self.flattened_size)

        # produce the mean and log-variance of the latent distribution
        self.mean = nn.Linear(self.flattened_size, latent_dim)
        self.variance = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x):
        # apply the convolutional layers
        x = self.layers(x)

        # dynamically calculate the size of the output from the last convolutional layer
        self.flattened_size = x.shape[1] * x.shape[2] * x.shape[3]
        # print(self.flattened_size)

        # flatten the output for dense layers
        x = x.view(-1, self.flattened_size)

        # compute the mean and log-variance of the latent space
        mean = self.mean(x)
        variance = self.variance(x)

        return mean, variance


# testing class VAEEncoder
# encoder = VAEEncoder(input_channels=3, latent_dim=100, max_dim=lr_train.max_dim)
# sample_tensors = []
# sample_results = {}
# print("Testing VAEEncoder...")
# for index in tqdm(range(1, 11), ncols=80):
#     # generate a tensor by index and add 1 as batch dimension to make it [1, 3, max_dim, max_dim]
#     tensor = lr_train.image_to_tensor(index)
#     sample_tensors.append(tensor)
#     tensor_mean, tensor_variance = encoder(tensor)
#     sample_results[index] = {
#         "mean": tensor_mean,
#         "variance": tensor_variance
#     }
#     # print(f"Mean: {t_mean}")
#     # print(f"Variance: {t_variance}")
# print("Testing successful.")


# define the Gaussian sampling function using mean and variance from latent space
def reparameterize(mean, variance):
    # σ = exp(0.5 * log(σ²))
    sigma = torch.exp(0.5 * variance)

    # ε ~ N(0, I)
    epsilon = torch.randn_like(sigma)

    # z = μ + εσ
    return mean + epsilon * sigma


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels, max_dim):
        # call the constructor of nn.Module
        nn.Module.__init__(self)

        # this should match the output shape of the encoder (15.9375 dim)
        self.start_dim = max_dim // 16

        # this should match the last convolutional layer of the encoder
        self.input_channels = 255

        # transform the latent data vector back to the shape suitable for convolutional layers
        self.dense_layer = nn.Linear(latent_dim, self.input_channels * self.start_dim * self.start_dim)

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.input_channels, 256, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, output_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=0),
            nn.Sigmoid()
        )

    def forward(self, z):
        # apply the convolutional layers
        z = self.dense_layer(z)
        z = z.view(-1, self.input_channels, self.start_dim, self.start_dim)
        x = self.layers(z)
        return x


# testing class VAEDecoder
# decoder = VAEDecoder(latent_dim=100, output_channels=3, max_dim=lr_train.max_dim)
# print(lr_train.max_dim)
# print("Testing VAEDecoder...")
# # getting x_tilda (new image tensor, shape [1, 3, 255, 255])
# for index in tqdm(range(1, 11), ncols=80):
#     # obtaining the distribution from mean and variance
#     mean = sample_results[index]["mean"]
#     variance = sample_results[index]["variance"]
#
#     z = reparameterize(mean, variance)
#
#     # reconstructing the data using the decoder
#     x_tilda = decoder(z)
#     print(x_tilda.shape)
#     # print(x_tilda)
# print("Testing successful.")


class VAELoss(nn.Module):
    def __init__(self):
        # call the constructor of nn.Module
        nn.Module.__init__(self)

        # load VGG16 and use only the first 23 layers for feature extraction
        self.extractor = vgg16(weights="VGG16_Weights.DEFAULT").features[:23].eval()
        self.extractor = nn.Sequential(self.extractor)

        # lock the parameters for objective judgement
        for param in self.extractor.parameters():
            param.requires_grad = False

        # define mse
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, x_tilda, x, mean, variance):
        # reconstruction loss
        reconstruction_loss = self.mse_loss(x_tilda, x)

        # KL divergence loss
        kl_divergence = -0.5 * torch.sum(1 + variance - mean.pow(2) - variance.exp())

        # perceptual loss using VGG16 features
        x_features = self.extractor(x)
        x_tilda_features = self.extractor(x_tilda)
        x_features = torch.nn.functional.normalize(x_features, dim = 1)
        x_tilda_features = torch.nn.functional.normalize(x_tilda_features, dim = 1)
        perceptual_loss = self.mse_loss(x_tilda_features, x_features)

        # Total loss
        total_loss = reconstruction_loss + kl_divergence + perceptual_loss

        return total_loss, reconstruction_loss, kl_divergence, perceptual_loss

# define training loop
latent_dim = 500
encoder = VAEEncoder(input_channels=3, latent_dim=latent_dim, max_dim=lr_train.max_dim).to(cpu_device)
decoder = VAEDecoder(latent_dim=latent_dim, output_channels=3, max_dim=lr_train.max_dim).to(cpu_device)
loss = VAELoss().to(cpu_device)

# define optimizer
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.005)

epochs = 10
batch_size = 1
total_loss_vals = []
recon_loss_vals = []
kl_loss_vals = []
perceptual_loss_vals = []

for epoch in range(epochs):
    print("-" * term_columns)
    print(f"Epoch {epoch + 1} begins...")

    # Initialize gradients
    optimizer.zero_grad()

    epoch_loss_vals = []
    for index in range(1, 801):  # 801
        # ground truth x from hr_train
        # x = hr_train.image_to_tensor(index).to(gpu_device)

        # compressed x from lr_train
        # x_compressed = lr_train.image_to_tensor(index).to(gpu_device)

        # Batch processing
        batch_indices = range(index, min(index + batch_size, 801))
        x_batch = []
        x_compressed_batch = []

        for i in batch_indices:
            # Load the high-resolution image and move it to CPU
            hr_image = hr_train.image_to_tensor(i).squeeze(0).to(cpu_device)
            x_batch.append(hr_image)

            # Load the compressed image and move it to GPU
            lr_image = lr_train.image_to_tensor(i).squeeze(0).to(cpu_device)
            x_compressed_batch.append(lr_image)

        x_batch = torch.stack(x_batch)
        x_compressed_batch = torch.stack(x_compressed_batch)

        # zero the optimizer
        optimizer.zero_grad()

        # print("Shape of x_compressed_batch:", x_compressed_batch.shape)

        # encoder data
        mean, variance = encoder(x_compressed_batch)

        # apply reparameterization
        z = reparameterize(mean, variance)

        # decode reconstructed data
        x_tilda = decoder(z)

        # calculate loss
        total_loss, reconstruction_loss, kl_divergence, perceptual_loss = loss(x_batch, x_tilda, mean, variance)
        epoch_loss_vals.append([total_loss.item(), reconstruction_loss.item(), kl_divergence.item(), perceptual_loss.item()])

        # backward pass and optimization step
        loss.backward()
        optimizer.step()

        # stats
        print(f"Tensor {index} - Loss: <total_loss> {total_loss}, <reconstruction_loss> {reconstruction_loss}, <kl_divergence> {kl_divergence}, <perceptual_loss> {perceptual_loss}")

        # collect garbage and empty CUDA cache
        gc.collect()
        torch.cuda.empty_cache()

    epoch_loss_vals = np.array(epoch_loss_vals)
    epoch_total_mean, epoch_recon_mean, epoch_kl_mean, epoch_perceptual_mean = epoch_loss_vals.mean(axis=0)
    total_loss_vals.append(epoch_total_mean)
    recon_loss_vals.append(epoch_recon_mean)
    kl_loss_vals.append(epoch_kl_mean)
    perceptual_loss_vals.append(epoch_perceptual_mean)

# plot the training results
plt.figure(figsize=(8,6))
plt.plot(np.arange(epochs), total_loss_vals, label="Total Loss", linestyle="-", linewidth=1, markersize=12)
plt.plot(np.arange(epochs), recon_loss_vals, label="Reconstruction Loss", linestyle="-", linewidth=1, markersize=12)
plt.plot(np.arange(epochs), kl_loss_vals, label="KL-Divergence Loss", linestyle="-", linewidth=1, markersize=12)
plt.plot(np.arange(epochs), perceptual_loss_vals, label="Perceptual Loss", linestyle="-", linewidth=1, markersize=12)
plt.xlabel("Epoch")
plt.legend()
plt.title("Training Results")
plt.show()

print("STATUS: Training is finished!")

# save the model
save_path = "trained_model.pth"
torch.save({
    'epoch': epoch,
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
}, save_path)