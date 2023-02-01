import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def mackey_glass(tau : int, n : int, N : int, x0: float=0.1, beta: float=0.2, gamma: float=0.1):

    """
    Generates the Mackey-Glass time series.
    
    Parameters:
    - tau (int): lag constant
    - n (int): number of time steps to generate
    - N (int): power on the denominator of the formula
    - x0 (float, optional): initial value of the time series
    - beta (float, optional): parameter for the nonlinear feedback
    - gamma (float, optional): parameter for the linear feedback
    
    Returns:
    - torch.Tensor: Mackey-Glass time series of size (n, 1)
    """

    t = np.arange(0, n)
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i-1] + beta*x[i- 1 -tau]/(1 + x[i-1-tau]**N) - gamma*x[i-1]
    return torch.tensor(x, dtype=torch.float32).reshape(-1, 1)


class MLP(nn.Module):
   
    def __init__(self, input_size: int, hidden_size_1: int, hidden_size_2: int, output_size : int):
        super(MLP, self).__init__()

        # Parameters

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        # Create layers and the activation layer
        self.input_layer = nn.Linear(self.input_size, self.hidden_size_1)
        self.hidden_layer_1 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.hidden_layer_2 = nn.Linear(self.hidden_size_2, self.hidden_size_2)
        self.activation_layer = nn.Sigmoid()

        # Create output layer
        self.output_layer = nn.Linear(self.hidden_size_2, output_size)

    def forward(self, x):

        # Function used to compute the forward pass

        # Compute first layer

        l1 = self.input_layer(x)
        l1 = self.activation_layer(l1)

        # Compute first hidden layer

        l1 = self.hidden_layer_1(l1)
        l1 = self.activation_layer(l1)

        # Compute second hidden layer
        
        l1 = self.hidden_layer_2(l1)
        l1 = self.activation_layer(l1)

        # Compute output layer

        out = self.output_layer(l1) # No need for activation function, it is already linear

        return out

class EarlyStopping:
    """
    A PyTorch implementation of early stopping.
    
    Args:
        patience (int, optional): Number of epochs to wait for improvement.
            Defaults to 7.
        delta (float, optional): Minimum change in the monitored value to qualify
            as an improvement. Defaults to 0.
        checkpoint_path (str, optional): Path to save the best model checkpoint.
            Defaults to 'checkpoint.pt'.
        is_maximize (bool, optional): Flag to indicate whether to maximize or minimize
            the monitored value. Defaults to True.
    """
    def __init__(self, patience=7, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.is_maximize = is_maximize
        
    def __call__(self, current_score, model):
        """
        Call method to update the early stopping status based on the current score.
        
        Args:
            current_score (float): The current score to compare against the best score.
            model (nn.Module): The PyTorch model to be saved if the score improves.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(current_score, model)
        elif self.is_maximize:
            if current_score < self.best_score + self.delta:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.save_checkpoint(current_score, model)
                self.counter = 0
        else:
            if current_score > self.best_score + self.delta:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = current_score
                self.save_checkpoint(current_score, model)
                self.counter = 0
    
    def save_checkpoint(self, score, model):
        """
        Saves the model checkpoint if the score improves.
        
        Args:
            score (float): The current score.
            model (nn.Module): The PyTorch model to be saved.
        """
        torch.save({
            'model_state_dict': model.state_dict(),
            'score': score
        }, self.checkpoint_path)

class RegularizedMLPTrainer:
    """
    A PyTorch trainer class for multi-layer perceptron models with penalization term and MSE loss.
    
    Args:
        model (nn.Module): The multi-layer perceptron model to be trained.
        dataloader (DataLoader): PyTorch dataloader to provide training data.
        p (float): The penalization coefficient.
        ds_val (TensorDataset): the input for the validation set
        loss_fn (nn.Module, optional): Loss function to be used for training. Defaults to nn.MSELoss().
        optimizer (torch.optim, optional): Optimization algorithm to be used for training. Defaults to torch.optim.SGD.
        patience (int, optional): Number of epochs to wait for improvement. Defaults to 7.
        delta (float, optional): Minimum change in the monitored value to qualify as an improvement. Defaults to 0.
        checkpoint_path (str, optional): Path to save the best model checkpoint. Defaults to 'checkpoint.pt'.
    """
    def __init__(self, model, train_dataloader, val_dataloader, loss_fn=nn.MSELoss(), optimizer=torch.optim.SGD, 
                 patience=7, delta=0, checkpoint_path='checkpoint.pt'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.early_stopping = EarlyStopping(patience=patience, delta=delta, 
                                            checkpoint_path=checkpoint_path)
        self.training_losses = []
        self.validation_losses = []

    def train(self, num_epochs):
        """
        Trains the multi-layer perceptron model for a specified number of epochs with penalization term.
        
        Args:
            num_epochs (int): The number of epochs to train the model for.
        """
        for epoch in range(num_epochs):
            training_loss = torch.Tensor([0.])
            for x, y in self.train_dataloader:
                outputs = self.model(x)
                reg_loss = self.loss_fn(outputs, y)
                # Add the penalization term to the loss
                self.optimizer.zero_grad()
                reg_loss.backward()
                self.optimizer.step()
                training_loss += reg_loss
            
            current_score = self.evaluate()
            self.training_losses.append(training_loss)
            self.validation_losses.append(current_score)
            self.early_stopping(current_score, self.model)
            if self.early_stopping.early_stop:
                # print(f'Early stopping at epoch {epoch + 1}')
                break

        
        return self.training_losses, self.validation_losses
                
    def evaluate(self):
        """
        Evaluates the model on the current data and returns the loss.
        
        Returns:
            Torch tensor: The evaluation loss.
        """
        validation_loss = torch.Tensor([0.])

        with torch.no_grad():
            for x, y in self.val_dataloader:
                    y_pred = self.model(x)
                    validation_loss += self.loss_fn(y_pred, y)
        return validation_loss

def plot_preds(title:str, y_pred, y) -> None:
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(y_pred.shape[0]) , y_pred, label='Prediction')
    ax.plot(np.arange(y_pred.shape[0]) , y, label='Ground truth')
    ax.set_xlabel("t")
    ax.set_ylabel("Mackey-Glass time series")
    ax.set_title(title)
    plt.legend()