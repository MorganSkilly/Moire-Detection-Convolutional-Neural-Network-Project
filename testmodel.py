

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

    