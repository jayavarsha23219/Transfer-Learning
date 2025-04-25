# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.

## DESIGN STEPS
### STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.
</br>

### STEP 2:
initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.
</br>

### STEP 3:
Train the model with training dataset.
<br/>
### STEP 4:
Evaluate the model with testing dataset.
<br/>
### STEP 5:
Make Predictions on New Data.
<br/>

## PROGRAM
```
## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:JAYAVARSHA T")
    print("Register Number:212223040075")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```
## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2025-04-25 112037](https://github.com/user-attachments/assets/3607994a-e54e-4247-b071-8d3fc29e1635)


### Confusion Matrix

![Screenshot 2025-04-25 112114](https://github.com/user-attachments/assets/f1d4f464-b6a2-402e-ba77-f2669e5c53e8)


### Classification Report

![Screenshot 2025-04-25 112140](https://github.com/user-attachments/assets/24a1919f-1415-4a0f-bcdd-5c2bc1456b50)


### New Sample Prediction

![Screenshot 2025-04-25 112229](https://github.com/user-attachments/assets/29b20d90-5374-432b-866c-24541c4b68f0)


## RESULT
Thus, the transfer Learning for classification using VGG-19 architecture has succesfully implemented.
