import os
import torch
from torch import nn
from torch import inference_mode
from torch.utils.data import DataLoader
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

if __name__ == "__main__":    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CPU_THREADS = os.cpu_count()

    # def walk_through_dir(dir_path):
    #     for dirpath, dirnames, filenames in os.walk(dir_path):
    #         print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
        
    DATASET_DIRECTORY = "Dataset"
    #walk_through_dir(DATASET_DIRECTORY)

    TRAIN_DIRECTORY = "Dataset/train/train/all"
    TEST_DIRECTORY = "Dataset/test_origin/test_origin"

    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

    SEED = 70
    NUM_EPOCHS = 30
    BATCH_SIZE = 10

    random.seed(SEED)
    seaborn.set_theme()

    # Define the transformations that should be applied to the images
    DATA_TRANSFORM = transforms.Compose([
        transforms.Resize(size = IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor() # Convert the image to a pytorch tensor
    ])


    print("\n#########################################\n")

    print(f"torch version: {torch.__version__}")
    print(f"compute device: {DEVICE}")
    print(f"dataset directory: {DATASET_DIRECTORY}")
    print(f"train directory: {TRAIN_DIRECTORY}")
    print(f"test directory: {TEST_DIRECTORY}")
    print(f"image size: {IMAGE_SIZE}")
    print(f"seed: {SEED}")
    print(f"epoch count: {NUM_EPOCHS}")
    print(f"batch size: {BATCH_SIZE}\n")
    print(f"data transform: {DATA_TRANSFORM}")

    print("\n#########################################")


    # def plot_transformed_images(image_paths, transform, n=3, seed=SEED):
    #     random.seed(seed)
    #     random_image_paths = random.sample(image_paths, k=n)
    #     for image_path in random_image_paths:
    #         with Image.open(image_path) as f:
    #             fig, ax = pyplot.subplots(1, 2)
    #             ax[0].imshow(f) 
    #             ax[0].set_title(f"Original \nSize: {f.size}")
    #             ax[0].axis("off")

    #             transformed_image = transform(f).permute(1, 2, 0)
    #             ax[1].imshow(transformed_image) 
    #             ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
    #             ax[1].axis("off")
    #             class_name = Path(image_path).stem.split('_')[1]
    #             fig.suptitle(f"Class: {class_name}", fontsize=16)

    #image_path_list= glob.glob(f"{DATASET_DIRECTORY}/*/*/*/*.jpg")
    #random_image_path = random.choice(image_path_list)
    #pyplot.ion()

    ## Display a random image
    #image_class = Path(random_image_path).parent.stem
    #img = Image.open(random_image_path)
    #print(f"Random image path: {random_image_path}")
    #print(f"Image class: {image_class}")
    #print(f"Image height: {img.height}") 
    #print(f"Image width: {img.width}")
    ##img.show()
    #img_as_array = numpy.asarray(img)
    #fig, ax = pyplot.subplots(figsize=(8, 6))
    #pyplot.imshow(img_as_array)
    #pyplot.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
    #pyplot.axis(False)
    #plot = pyplot.show()
    #input("Press Enter to close the plot...")
    #pyplot.close(fig)

    #plot_transformed_images(image_path_list, transform=DATA_TRANSFORM, n=3)

    train_data = datasets.ImageFolder(root=TRAIN_DIRECTORY,
                                    transform=DATA_TRANSFORM,
                                    target_transform=None)

    test_data = datasets.ImageFolder(root=TEST_DIRECTORY, transform=DATA_TRANSFORM)

    print(f"\ntrain data:\n{train_data}\ntest data:\n{test_data}")

    class_names = train_data.classes
    print("\nclass names: ",class_names)

    class_dict = train_data.class_to_idx
    print("\nclass names as a dict: ",class_dict)

    print("\ndata set lengths: ", len(train_data), len(test_data))

    print("\n#########################################")

    #----------------------
    #img, label = train_data[0][0], train_data[0][1]
    #print(f"Image tensor:\n{img}")
    #print(f"Image shape: {img.shape}")
    #print(f"Image datatype: {img.dtype}")
    #print(f"Image label: {label}")
    #print(f"Label datatype: {type(label)}")

    #img_permute = img.permute(1, 2, 0)
    #print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    #print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

    # Plot the image
    #pyplot.figure(figsize=(10, 7))
    #pyplot.imshow(img.permute(1, 2, 0))
    #pyplot.axis("off")
    #pyplot.title(class_names[label], fontsize=14)
    #---------------------


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

    #img, label = next(iter(train_dataloader))

    #print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    #print(f"Label shape: {label.shape}")

    # Create training transform with TrivialAugment
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()])
    
    print(f"\ntraining transform: {train_transform}")

    # Create testing transform (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()])
    
    print(f"\ntesting transform: {test_transform}")
    
    print("\n#########################################")

    train_data_augmented = datasets.ImageFolder(TRAIN_DIRECTORY, transform=train_transform)
        
    print(f"\ntraining augmented: {train_data_augmented}")

    test_data_augmented = datasets.ImageFolder(TEST_DIRECTORY, transform=test_transform)

    print(f"\ntesting augmented: {train_data_augmented}")
    
    print("\n#########################################")

    torch.manual_seed(SEED)

    train_dataloader_augmented = DataLoader(train_data_augmented, 
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True,
                                            num_workers=CPU_THREADS)
    
    print(f"\ntraining dataloader augmented: {train_dataloader_augmented}")

    test_dataloader_augmented = DataLoader(test_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=False, 
                                        num_workers=CPU_THREADS)
    
    print(f"\ntesting dataloader augmented: {test_dataloader_augmented}")
    
    print("\n#########################################")

    class ImageClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layer_1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2)
            )
            self.conv_layer_2 = nn.Sequential(
                nn.Conv2d(64, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(2)
            )
            self.conv_layer_3 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(2)
            )
            
            # Use a placeholder for in_features that will be dynamically calculated
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=1, out_features=2)  # Replace 1 with the correct value dynamically
            )

        def forward(self, x: torch.Tensor):
            x = self.conv_layer_1(x)
            x = self.conv_layer_2(x)
            x = self.conv_layer_3(x)
            x = self.classifier(x)
            return x

        def compute_linear_input_size(self, input_shape):
            # Pass a dummy tensor through the convolutional layers
            dummy_input = torch.randn(1, *input_shape)
            x = self.conv_layer_1(dummy_input)
            x = self.conv_layer_2(x)
            x = self.conv_layer_3(x)
            return x.numel()

    # Instantiate the model
    model = ImageClassifier()
   
    print(f"\nmodel: {model}")

    # Compute the correct input size for the Linear layer
    input_shape = (3, IMAGE_HEIGHT, IMAGE_WIDTH)  # Assuming 3 channels (RGB)
    
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

    # 1. Get a batch of images and labels from the DataLoader
    #img_batch, label_batch = next(iter(train_dataloader_augmented))

    # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    # img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    # print(f"Single image shape: {img_single.shape}\n")

    # 3. Perform a forward pass on a single image
    # model.eval()
    # pred = inference_mode.model(img_single.to(DEVICE))
        
    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    # print(f"Output logits:\n{pred}\n")
    # print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    # print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    # print(f"Actual label:\n{label_single}")
        
    # do a test pass through of an example input size 
    summary(model, input_size=[1, 3, IMAGE_WIDTH ,IMAGE_HEIGHT])
    print("\n#########################################\n")

    def train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer):
        # Put model in train mode
        model.train()
        print("\nTraining model...")        
        
        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0
        
        # Loop through data loader data batches
        for batch, (X, y) in enumerate(dataloader):
            print(f"Training on tensor: {batch}")

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
        return train_loss, train_acc

    def test_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module):
        # Put model in eval mode
        model.eval() 
        print("\nEvaluating model...")  
        
        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        
        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(dataloader):
                print(f"Evaluating on tensor: {batch}")
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
        for epoch in range(epochs):
            print(f"\nTraining epoch: {epoch}")
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
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # 6. Return the filled results at the end of the epochs
        return results

    # Set random seeds
    torch.manual_seed(SEED) 
    torch.cuda.manual_seed(SEED)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    # Train model_0 
    model_results = train(model=model,
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_augmented,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

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