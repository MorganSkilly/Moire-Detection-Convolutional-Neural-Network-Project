import os
import torch
from torch import nn
from torch import inference_mode
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchinfo import summary
import random
from PIL import Image
import glob
from pathlib import Path
import numpy
import matplotlib.pyplot as pyplot
import seaborn
import time
import torchinfo
import classifier
import parameters
from tqdm.auto import tqdm
from timeit import default_timer as timer
import trainingsteps
  
def train(hyperparameters, directories): 

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU_THREADS = os.cpu_count()

    hyperparameters.print_hyperparameters()
    directories.print_directories()

    random.seed(hyperparameters.SEED)
    torch.manual_seed(hyperparameters.SEED)
    seaborn.set_theme()

    DATA_TRANSFORM = transforms.Compose([
        transforms.Resize(size = hyperparameters.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root = directories.TRAIN_DIRECTORY, transform = DATA_TRANSFORM)
    test_data = datasets.ImageFolder(root = directories.TEST_DIRECTORY, transform=DATA_TRANSFORM)

    class_names = train_data.classes
    class_dict = train_data.class_to_idx

    print("\nclass names: ",class_dict)
    print("\ndata set lengths: ", len(train_data), len(test_data))

    # Create training transform with TrivialAugment
    train_transform = transforms.Compose([
        transforms.Resize(hyperparameters.IMAGE_SIZE),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()])    
    train_data_augmented = datasets.ImageFolder(directories.TRAIN_DIRECTORY,
                                                transform=train_transform)
    train_dataloader_augmented = DataLoader(train_data_augmented, 
                                            batch_size=hyperparameters.BATCH_SIZE, 
                                            shuffle=True,
                                            num_workers=CPU_THREADS)
    
    # Create testing transform (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(hyperparameters.IMAGE_SIZE),
        transforms.ToTensor()])
    test_data_augmented = datasets.ImageFolder(directories.TEST_DIRECTORY,
                                               transform=test_transform)
    test_dataloader_augmented = DataLoader(test_data_augmented, 
                                        batch_size=hyperparameters.BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=CPU_THREADS)
   
    print(f"\ntraining dataloader augmented: {train_dataloader_augmented}")    
    print(f"\ntesting dataloader augmented: {test_dataloader_augmented}")

    # Instantiate the model
    model = classifier.ImageClassifier()
    model = model.to(DEVICE)
    print(f"\nmodel: {model}")

    # 1. Take in various parameters required for training and test steps
    def train(model: torch.nn.Module, 
            train_dataloader: torch.utils.data.DataLoader, 
            test_dataloader: torch.utils.data.DataLoader, 
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
            epochs: int = 5):
        
        # 2. Create empty results dictionary
        results = {"train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        
        # 3. Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            print(f"\n\n\nTraining epoch: {epoch}")

            train_loss, train_acc = trainingsteps.trainbatch(model=model,
                                                            dataloader=train_dataloader,
                                                            loss_fn=loss_fn,
                                                            optimizer=optimizer,
                                                            timer=timer,
                                                            DEVICE=DEVICE)
            test_loss, test_acc = trainingsteps.testbatch(model=model,
                                                        dataloader=test_dataloader,
                                                        loss_fn=loss_fn,
                                                        timer=timer,
                                                        DEVICE=DEVICE)
            
            # 4. Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"training loss: {train_loss:.4f} | "
                f"training accuracy: {train_acc:.4f} | "
                f"testing loss: {test_loss:.4f} | "
                f"test accuracy: {test_acc:.4f}"
            )

            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # 6. Return the filled results at the end of the epochs
        return results

    # Set random seeds
    torch.manual_seed(hyperparameters.SEED) 
    torch.cuda.manual_seed(hyperparameters.SEED)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    # Start the timer
    start_time = timer()

    # Train model_0 
    model_results = train(model=model,
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_augmented,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=hyperparameters.NUM_EPOCHS)
    
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

    folder = "models"
    base_filename = "moire_model.pth"
    counter = 1

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Create the full file path
    file_path = os.path.join(folder, base_filename)

    # Check if the file exists and increment the counter if needed
    while os.path.exists(file_path):
        file_path = os.path.join(folder, f"moire_model_{counter}.pth")
        counter += 1

    # Save the model
    torch.save(model.state_dict(), file_path)

    def plot_loss_curves(results, counter):
        results = dict(list(results.items()))

        # Get the loss values of the results dictionary (training and test)
        loss = results['train_loss']
        test_loss = results['test_loss']

        # Get the accuracy values of the results dictionary (training and test)
        accuracy = results['train_acc']
        test_accuracy = results['test_acc']

        # Figure out how many epochs there were
        epochs = range(len(results['train_loss']))

        # Setup a plot 
        pyplot.figure(figsize=(15, 7))

        # Plot loss
        pyplot.subplot(1, 2, 1)
        pyplot.plot(epochs, loss, label='train_loss')
        pyplot.plot(epochs, test_loss, label='test_loss')
        pyplot.title('Loss')
        pyplot.xlabel('Epochs')
        pyplot.legend()

        # Plot accuracy
        pyplot.subplot(1, 2, 2)
        pyplot.plot(epochs, accuracy, label='train_accuracy')
        pyplot.plot(epochs, test_accuracy, label='test_accuracy')
        pyplot.title('Accuracy')
        pyplot.xlabel('Epochs')
        pyplot.legend()

        # Ensure the 'models' folder exists
        os.makedirs("models", exist_ok=True)

        # Define the save path with a unique counter in the filename
        save_path = os.path.join("models", f"loss_accuracy_plot_{str(counter - 1).zfill(3)}.png")

        # Save the plot to the models folder with the unique filename
        pyplot.savefig(save_path)

        # Optionally close the plot to free memory
        pyplot.close()

        print(f"Plot saved to: {save_path}")

    plot_loss_curves(model_results, counter)
    
    # Choose a image.
    custom_image_path = "Dataset/test_origin/test_origin/moire/0000_moire.jpg"

    # Load in custom image and convert the tensor values to float32
    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    custom_image = custom_image / 255. 

    # Print out image data
    print(f"Custom image tensor:\n{custom_image}\n")
    print(f"Custom image shape: {custom_image.shape}\n")
    print(f"Custom image dtype: {custom_image.dtype}")

    custom_image_transform = transforms.Compose([
        transforms.Resize(hyperparameters.IMAGE_SIZE),
    ])

    # Transform target image
    custom_image_transformed = custom_image_transform(custom_image)

    # Print out original shape and new shape
    print(f"Original shape: {custom_image.shape}")
    print(f"New shape: {custom_image_transformed.shape}")

    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to image
        custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
        
        # Print out different shapes
        print(f"Custom image transformed shape: {custom_image_transformed.shape}")
        print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")
        
        # Make a prediction on image with an extra dimension
        custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(DEVICE))

        # Let's convert them from logits -> prediction probabilities -> prediction labels
    # Print out prediction logits
    print(f"Prediction logits: {custom_image_pred}")

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    print(f"Prediction probabilities: {custom_image_pred_probs}")

    # Convert prediction probabilities -> prediction labels
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
    print(f"Prediction label: {custom_image_pred_label}")

    # put pred label to CPU, otherwise will error
    custom_image_pred_class = class_names[custom_image_pred_label.cpu()]

    # Plot custom image
    pyplot.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
    pyplot.title(f"Image shape: {custom_image.shape} Class: {custom_image_pred_class}")
    pyplot.axis(False)