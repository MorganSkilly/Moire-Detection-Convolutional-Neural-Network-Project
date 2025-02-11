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

if __name__ == "__main__":    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU_THREADS = os.cpu_count()

    hyperparameters = parameters.Hyperparameters()
    hyperparameters.print_hyperparameters()
    directories = parameters.Directories()
    directories.print_directories()


    random.seed(hyperparameters.SEED)
    seaborn.set_theme()

    # Define the transformations that should be applied to the images
    DATA_TRANSFORM = transforms.Compose([
        transforms.Resize(size = hyperparameters.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor() # Convert the image to a pytorch tensor
    ])


    print("\n=============================================")
    print(f"torch version: {torch.__version__}")
    print(f"compute device: {DEVICE}")
    print(f"dataset directory: {directories.DATASET_DIRECTORY}")
    print(f"train directory: {directories.TRAIN_DIRECTORY}")
    print(f"test directory: {directories.TEST_DIRECTORY}")
    print(f"image size: {hyperparameters.IMAGE_SIZE}")
    print(f"seed: {hyperparameters.SEED}")
    print(f"epoch count: {hyperparameters.NUM_EPOCHS}")
    print(f"batch size: {hyperparameters.BATCH_SIZE}\n")
    print("\n=============================================\n")

    train_data = datasets.ImageFolder(root=directories.TRAIN_DIRECTORY,
                                    transform=DATA_TRANSFORM,
                                    target_transform=None)

    test_data = datasets.ImageFolder(root=directories.TEST_DIRECTORY, transform=DATA_TRANSFORM)

    print(f"\ntrain data:\n{train_data}\ntest data:\n{test_data}")

    class_names = train_data.classes
    print("\nclass names: ",class_names)

    class_dict = train_data.class_to_idx
    print("\nclass names as a dict: ",class_dict)

    print("\ndata set lengths: ", len(train_data), len(test_data))

    print("\n#########################################")

    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=1, # how many samples per batch?
                                num_workers=CPU_THREADS,
                                shuffle=True) # shuffle the data?
    
    print(f"\ntraining dataloader: {train_dataloader}")

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=1, 
                                num_workers=CPU_THREADS, 
                                shuffle=False) # don't usually need to shuffle testing data
    
    print(f"\ntesting dataloader: {train_dataloader}")
    
    print("\n#########################################")

    # Create training transform with TrivialAugment
    train_transform = transforms.Compose([
        transforms.Resize(hyperparameters.IMAGE_SIZE),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()])
    
    print(f"\ntraining transform: {train_transform}")

    # Create testing transform (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(hyperparameters.IMAGE_SIZE),
        transforms.ToTensor()])
    
    print(f"\ntesting transform: {test_transform}")
    
    print("\n#########################################")

    train_data_augmented = datasets.ImageFolder(directories.TRAIN_DIRECTORY, transform=train_transform)
        
    print(f"\ntraining augmented: {train_data_augmented}")

    test_data_augmented = datasets.ImageFolder(directories.TEST_DIRECTORY, transform=test_transform)

    print(f"\ntesting augmented: {train_data_augmented}")
    
    print("\n#########################################")

    torch.manual_seed(hyperparameters.SEED)

    train_dataloader_augmented = DataLoader(train_data_augmented, 
                                            batch_size=hyperparameters.BATCH_SIZE, 
                                            shuffle=True,
                                            num_workers=CPU_THREADS)
    
    print(f"\ntraining dataloader augmented: {train_dataloader_augmented}")

    test_dataloader_augmented = DataLoader(test_data_augmented, 
                                        batch_size=hyperparameters.BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=CPU_THREADS)
    
    print(f"\ntesting dataloader augmented: {test_dataloader_augmented}")
    
    print("\n#########################################")

    # Instantiate the model
    model = classifier.ImageClassifier()
   
    print(f"\nmodel: {model}")

    # Compute the correct input size for the Linear layer
    input_shape = (3, hyperparameters.IMAGE_HEIGHT, hyperparameters.IMAGE_WIDTH)  # Assuming 3 channels (RGB)
    
    print(f"\ninput shape: {input_shape}")

    in_features = model.compute_linear_input_size(input_shape)

    print(f"\ninput features: {in_features}")

    # Update the Linear layer
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=in_features, out_features=2)  # Use the correct in_features
    )

    model = model.to(DEVICE)

    print(f"\nmodel: {model}")

    print("\n#########################################\n")
        
    # do a test pass through of an example input size 
    summary(model, input_size=[1, 3, hyperparameters.IMAGE_WIDTH ,hyperparameters.IMAGE_HEIGHT])
    print("\n#########################################\n")

    def train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer):
        # Put model in train mode
        model.train()
        print("\nTraining model...")
        
        # Start the timer
        start_time = timer()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0
        
        # Loop through data loader data batches
        for batch, (X, y) in tqdm(enumerate(dataloader)):

            # Send data to target device
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item() 

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)      

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        end_time = timer()
        print(f"Batch training time: {end_time-start_time:.3f} seconds")

        return train_loss, train_acc

    def test_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module):
        # Put model in eval mode
        model.eval() 
        print("\nEvaluating model...")  
        
        # Start the timer
        start_time = timer()
        
        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        
        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in tqdm(enumerate(dataloader)):
                # Send data to target device
                X, y = X.to(DEVICE), y.to(DEVICE)
        
                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                
                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
                
        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

        end_time = timer()
        print(f"Batch evaluating time: {end_time-start_time:.3f} seconds")

        return test_loss, test_acc

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
            train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer)
            test_loss, test_acc = test_step(model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn)
            
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

    directory = "Dataset/test_origin/test_origin/clean"

    moire = 0
    clean = 0

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        
        # Check if it is a file (not a directory)
        if os.path.isfile(file_path):
            print(f"Found file: {filename}")

        custom_image_path = file_path

        # Load in custom image and convert the tensor values to float32
        custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

        # Divide the image pixel values by 255 to get them between [0, 1]
        custom_image = custom_image / 255. 

        custom_image_transform = transforms.Compose([
            transforms.Resize(hyperparameters.IMAGE_SIZE),
        ])

        # Transform target image
        custom_image_transformed = custom_image_transform(custom_image)

        model.eval()
        with torch.inference_mode():
            # Add an extra dimension to image
            custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
            
            # Make a prediction on image with an extra dimension
            custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(DEVICE))

            # Let's convert them from logits -> prediction probabilities -> prediction labels

        # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)

        # Convert prediction probabilities -> prediction labels
        custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)

        custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
        custom_image_pred_class

        if custom_image_pred_class == "moire":
            moire += 1
        else:
            clean += 1

        print(f"moire: {moire} clean: {clean}")

        directory = "Dataset/test_origin/test_origin/moire"

    moire = 0
    clean = 0

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        
        # Check if it is a file (not a directory)
        if os.path.isfile(file_path):
            print(f"Found file: {filename}")

        custom_image_path = file_path

        # Load in custom image and convert the tensor values to float32
        custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

        # Divide the image pixel values by 255 to get them between [0, 1]
        custom_image = custom_image / 255. 

        custom_image_transform = transforms.Compose([
            transforms.Resize(hyperparameters.IMAGE_SIZE),
        ])

        # Transform target image
        custom_image_transformed = custom_image_transform(custom_image)

        model.eval()
        with torch.inference_mode():
            # Add an extra dimension to image
            custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
            
            # Make a prediction on image with an extra dimension
            custom_image_pred = model(custom_image_transformed.unsqueeze(dim=0).to(DEVICE))

            # Let's convert them from logits -> prediction probabilities -> prediction labels

        # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)

        # Convert prediction probabilities -> prediction labels
        custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)

        custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
        custom_image_pred_class

        if custom_image_pred_class == "moire":
            moire += 1
        else:
            clean += 1

        print(f"moire: {moire} clean: {clean}")