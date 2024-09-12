# Type
from typing import List, Optional, Callable, Type, Union
# Neural network
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
# Trainer
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
# Helpers
from collections import OrderedDict
import matplotlib.pyplot as plt
from time import time
# from torch.utils.tensorboard import SummaryWriter


class DeepNeuralNetwork(nn.Module):
    def __init__(
        self, 
        layers: List[int], 
        activation_fn: Type[nn.Module] = nn.ReLU, 
        output_activation_fn: Optional[Type[nn.Module]] = None, 
        init_method: Optional[Callable[[nn.Module], None]] = None, 
        output_init_method: Optional[Callable[[nn.Module], None]] = None, 
        regularization: Optional[Type[nn.Module]] = None, 
        dropout_p: Optional[float] = None
    ):
        """
        Initialize the Deep Neural Network.

        Args:
            layers (List[int]): Ordered list of the number of neurons per layer.
            activation_fn (Type[nn.Module]): Activation function class from torch.nn (default: nn.ReLU).
            output_activation_fn (Optional[Type[nn.Module]]): Activation function class from torch.nn for the output layer.
            init_method (Optional[Callable[[nn.Module], None]]): Method for weights initialization.
            output_init_method (Optional[Callable[[nn.Module], None]]): Method for weights initialization on the last layer.
            regularization (Optional[Type[nn.Module]]): Regularization layer (e.g., nn.Dropout).
            dropout_p (Optional[float]): Dropout probability for regularization.

        Note:
            init_method should be a function that takes a layer as input and applies the initialization.
            
        Example:
            dnn = DeepNeuralNetwork(
                layers=[784, 128, 64, 10], 
                activation_fn=nn.ReLU,
                output_activation_fn=nn.ReLU,
                init_method=torch.nn.init.xavier_uniform_,
                output_init_method=torch.nn.init.xavier_uniform_,
                regularization=nn.Dropout,
                dropout_p=0.5
            )
        """
        super(DeepNeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            self._initialize_topology(
                layers,
                activation_fn,
                output_activation_fn,
                init_method,
                output_init_method,
                regularization,
                dropout_p
            )
        )

    def forward(self, x):
        """Define the forward pass of the network."""
        return self.layers(x)

    def _initialize_topology(
        self, 
        layers: List[int], 
        activation_fn: Type[nn.Module], 
        output_activation_fn: Optional[Type[nn.Module]], 
        init_method: Optional[Callable[[torch.Tensor], None]], 
        output_init_method: Optional[Callable[[torch.Tensor], None]], 
        regularization: Optional[Type[nn.Module]], 
        dropout_p: Optional[float]
    ) -> OrderedDict:
        """
        Initialize the layers and topology of the neural network.

        Args:
            layers (List[int]): Ordered list of the number of neurons per layer.
            activation_fn (Type[nn.Module]): Activation function class for hidden layers.
            output_activation_fn (Optional[Type[nn.Module]]): Activation function class for the output layer.
            init_method (Optional[Callable[[torch.Tensor], None]]): Method to initialize weights of hidden layers.
            output_init_method (Optional[Callable[[torch.Tensor], None]]): Method to initialize weights of the output layer.
            regularization (Optional[Type[nn.Module]]): Regularization method, e.g., `nn.Dropout`.
            dropout_p (Optional[float]): Dropout probability for regularization.

        Returns:
            OrderedDict: An ordered dictionary containing the layers, activations, and regularization.
        """
        depth = len(layers) - 1
        layers_list = []
        # All layers except output layer
        for i in range(depth):
            layer = nn.Linear(layers[i], layers[i + 1])
            
            # Set up for all but the last layer
            if i < depth - 1:  
                # Apply general initialization if specified
                if init_method is not None:
                    init_method(layer.weight)
                layers_list.append((f'layer_{i}', layer))
                # Add activation function
                layers_list.append((f'activation_{i}', activation_fn()))
                # Add regularization if specified
                if regularization and dropout_p:
                    layers_list.append((f'dropout_{i}', regularization(dropout_p)))
                    
            # Special handling for the last layer
            else:
                # Apply different initialization
                if output_init_method is not None:
                    output_init_method(layer.weight)  
                layers_list.append((f'layer_{i}', layer))
                # Add output activation function if specified
                if output_activation_fn is not None:
                    layers_list.append((f'activation_{i}', output_activation_fn()))
        
        return OrderedDict(layers_list)
    
class ModelTrainer:
    def __init__(self, model, X, y, batch_size=32, epochs=10, optimizer='Adam', lr=0.001, loss_fn='cross_entropy', validation_split=0.2, test_split=0.1, experiment_name='default_experiment', lr_scheduler=None, precision_optimization = False):
        """
        Initialize the Model Trainer with hyperparameters, and dataset, and automatically split the dataset into training, validation, and test sets.

        Args:
            model (DeepNeuralNetwork): Instance of the DeepNeuralNetwork.
            X (torch.Tensor): Input data which will be split into training, validation, and test sets.
            y (torch.Tensor): Targets/labels which will be split into training, validation, and test sets.
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs to train the model.
            optimizer (str): Name of the optimizer to use ('Adam', 'SGD', 'RMSprop').
            lr (float): Learning rate.
            loss_fn (str): Loss function ('cross_entropy' for classification, 'mse' for regression).
            validation_split (float): Fraction of the dataset to use for validation.
            test_split (float): Fraction of the dataset to use for testing.
        
        Exmaple:
            trainer = ModelTrainer(
                model=model,
                X=X_tensor,
                y=y_tensor,
                batch_size=64,
                epochs=50,
                optimizer='Adam',
                lr=0.001,
                loss_fn='mse', # Depends on model task (regression or classification)
                validation_split=0.2,
                test_split=0.1,
                experiment_name='Problem 1'
            )
            trainer.run()
            
        Note:
            Despite the loss functions, there will be computes MAE for regression tasks as metric and Accuracy for classification.
        """
        self.model = model if not precision_optimization else model.half() # Convert model to use float16
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_name = optimizer
        self.lr = lr
        self.loss_fn_name = loss_fn
        self.validation_split = validation_split
        self.test_split = test_split
        
        # Available Loss Functions per task
        self.available_loss_fns = {
            'regression': {
                'mse':nn.MSELoss(),
            },
            'classification': {
                'cross_entropy':nn.CrossEntropyLoss(),
            }
        }
        
        # Logs
        self.avg_loss_log = []
        
        # TensorBoard Writer
        self.experiment_name = experiment_name
        # self.writer = SummaryWriter(f'runs/{experiment_name}')
        
        
        
        self._prepare_data()
        self._set_optimizer()
        self._set_loss_function()

        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)  if lr_scheduler is None else lr_scheduler
    #
    # Helpers
    #
    def _prepare_data(self):
        """Prepares training, validation, and test data loaders from the dataset."""
        dataset = TensorDataset(self.X, self.y)
        # Calculate sizes for validation and test sets
        total_size = len(dataset)
        test_size = int(total_size * self.test_split)
        validation_size = int(total_size * self.validation_split)
        train_size = total_size - validation_size - test_size
        
        # Ensure that we have a valid split
        if train_size <= 0:
            raise ValueError("Training set size is non-positive, adjust validation and test split fractions.")
        
        # Split dataset
        train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

        # Prepare data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def _set_optimizer(self):
        """Configures the optimizer."""
        optimizers = {
            'Adam': Adam(self.model.parameters(), lr=self.lr),
            'SGD': SGD(self.model.parameters(), lr=self.lr),
            # Add optimizers
        }
        self.optimizer = optimizers[self.optimizer_name]

    def _set_loss_function(self):
        """Sets the loss function based on the task."""
        
        regression_loss_functions = self.available_loss_fns['regression']
        classification_loss_functions = self.available_loss_fns['classification']
        
        if self.loss_fn_name in regression_loss_functions.keys():
            self.loss_fn = regression_loss_functions[self.loss_fn_name]
            
        elif self.loss_fn_name in classification_loss_functions.keys():
            self.loss_fn = classification_loss_functions[self.loss_fn_name]
        else:
            raise ValueError(f"Unsupported loss function. Choose {regression_loss_functions.keys()+classification_loss_functions.keys()}.")

    def _clean_inputs_and_targets_datatype(self, inputs, targets):
        # print('inputs size: ',inputs.shape, inputs.type) 
        # print('targets size: ',targets.shape, targets.type) 
        if not self.loss_fn_name in self.available_loss_fns['classification'].keys():
            return  inputs, targets
        return inputs.float(), targets.long().squeeze()

    #
    # Metrics
    #
    def _compute_accuracy(self, outputs, targets):
        """Computes accuracy for classification tasks."""
        _, predictions = outputs.max(1)
        correct = predictions.eq(targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
        return accuracy

    def _compute_mae(self, outputs, targets):
        """Computes mean absolute error for regression tasks."""
        absolute_errors = (outputs - targets).abs()
        mae = absolute_errors.mean().item()
        return mae
    
        
    #
    # Main
    #
    def _evaluate(self, data_loader):
        """Evaluates the model on the given data loader and computes metrics."""
        self.model.eval() # Set model on testing mode
        total_loss = 0
        total_accuracy = 0
        total_mae = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = self._clean_inputs_and_targets_datatype(inputs, targets)
                outputs = self.model(inputs).float()  # Convert model outputs to float32 for loss computation (in case of optimized inputs)
                loss = self.loss_fn(outputs, targets.float())
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Compute metrics based on the loss function
                if self.loss_fn_name in self.available_loss_fns['classification'].keys():
                    total_accuracy += self._compute_accuracy(outputs, targets) * batch_size
                elif self.loss_fn_name in self.available_loss_fns['regression'].keys():
                    total_mae += self._compute_mae(outputs, targets) * batch_size

        self.model_outputs = outputs # for postprocessing (?) it may be useful
        # Compute average metrics
        avg_loss = total_loss / total_samples
        metrics = {'avg_loss': avg_loss}

        if self.loss_fn_name in self.available_loss_fns['classification'].keys():
            avg_accuracy = total_accuracy / total_samples * 100 # Percentage
            metrics['avg_accuracy'] = avg_accuracy
        elif self.loss_fn_name in self.available_loss_fns['regression'].keys():
            avg_mae = total_mae / total_samples
            metrics['avg_mae'] = avg_mae

        return metrics
        
    def train(self):
        """Trains the model and computes metrics for each epoch."""
        for epoch in range(self.epochs):
            self.model.train() # Set model on training mode
            
            total_loss = 0
            total_accuracy = 0
            total_mae = 0
            total_samples = 0

            for inputs, targets in self.train_loader:
                inputs, targets = self._clean_inputs_and_targets_datatype(inputs, targets)
                self.optimizer.zero_grad()
                outputs = self.model(inputs).float()  # Convert model outputs to float32 for loss computation (in case of optimized inputs)
                # print('outputs: ', outputs, 'inputs:', inputs)
                loss = self.loss_fn(outputs, targets.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                # Batch metrics
                if self.loss_fn_name in self.available_loss_fns['classification'].keys():
                    total_accuracy += self._compute_accuracy(outputs, targets) * inputs.size(0)
                elif self.loss_fn_name in self.available_loss_fns['regression'].keys():
                    total_mae += self._compute_mae(outputs, targets) * inputs.size(0)
            
            # After completing the epoch, update the learning rate
            self.scheduler.step()
            
            # Metrics 
            avg_loss = total_loss / total_samples
            self.avg_loss_log.append(avg_loss)
            # self.writer.add_scalar('Loss/Train', avg_loss, epoch)
            
            if self.loss_fn_name in self.available_loss_fns['classification'].keys():
                avg_accuracy = total_accuracy / total_samples * 100 # Percentage
                print(f'\tEpoch {epoch+1}/{self.epochs}\n\t\tLoss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}%')
                # self.writer.add_scalar('Accuracy/train', avg_accuracy, epoch)

            elif self.loss_fn_name in self.available_loss_fns['regression'].keys():
                avg_mae = total_mae / total_samples
                print(f'\tEpoch {epoch+1}/{self.epochs}\n\t\tLoss: {avg_loss:.4f}, MAE: {avg_mae:.4f}')
                # self.writer.add_scalar('MAE/train', avg_mae, epoch)
            
            # Validate
            self._validate()

    def _validate(self):
        """Validates the model and computes additional metrics with weighted averages."""
        self.model.eval() # Set model on testing mode
        
        metrics = self._evaluate(self.validation_loader)
        # self.writer.add_scalar('Loss/Validation', metrics["avg_loss"], epoch)
        print(f'\t\tValidation Loss: {metrics["avg_loss"]:.4f}')
        
        if 'avg_accuracy' in metrics:
            # self.writer.add_scalar('Accuracy/Validation', metrics["avg_accuracy"], epoch)
            print(f'\t\tValidation Accuracy: {metrics["avg_accuracy"]:.4f}%')
        if 'avg_mae' in metrics:
            # self.writer.add_scalar('MAE/Validation', metrics["avg_mae"], epoch)
            print(f'\t\tValidation MAE: {metrics["avg_mae"]:.4f}')
            
    def test(self):
        """Tests the model on a new dataset and computes metrics."""
        self.model.eval() # Set model on testing mode
        
        metrics = self._evaluate(self.test_loader)
        # self.writer.add_scalar('Loss/Test', metrics["avg_loss"], epoch)
        print(f'\tTest Loss: {metrics["avg_loss"]:.4f}')
        
        if 'avg_accuracy' in metrics:
            # self.writer.add_scalar('Accuracy/Test', metrics["avg_accuracy"], epoch)
            print(f'\tTest Accuracy: {metrics["avg_accuracy"]:.4f}%')
        if 'avg_mae' in metrics:
            # self.writer.add_scalar('MAE/Test', metrics["avg_mae"], epoch)
            print(f'\tTest MAE: {metrics["avg_mae"]:.4f}')

    #
    # Tensor Board
    #
    # def close(self):
    #     """Close the TensorBoard writer."""
    #     self.writer.close()
    
    def run(self):
        start = time()
        print(f'{self.experiment_name.capitalize()} Training...')
        self.train()
        print(f'{self.experiment_name.capitalize()} Testing...')
        self.test()
        print(f'\n{self.experiment_name.capitalize()} RunTime: {time()-start} sec')
        self.plot_training_loss()
        # Close the TensorBoard writer
        # self.close()   

    #
    # Extras
    #
    def predict(self, X):
        self.model.eval()
        
        X_tensor = torch.from_numpy(X.astype('float32'))
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.numpy()
    
    def plot_training_loss(self):
        plt.plot(self.avg_loss_log, 'r', label='Avg Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.experiment_name}: Average Training Loss')
        plt.grid(visible=True, alpha=0.33)
        plt.legend()
        plt.show()
