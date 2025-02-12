import torch
from tqdm.auto import tqdm
from torch import nn

def trainbatch(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                timer,
                DEVICE):
        
        model.train()
        print("\nTraining model...")
        
        # Start the timer
        start_time = timer()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0
        
        # Loop through data loader data batches
        for batch, (X, y) in tqdm(enumerate(dataloader)):
            print(batch)
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

def testbatch(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
                timer,
                DEVICE):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
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