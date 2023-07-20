# model code for simple convolutional nerual network followed by fully connected layers
import numpy as np
from matplotlib import pyplot as plt
import time
import copy

import torch
import torch.nn as nn

class ClassifierModel(nn.Module):
    # constructor
    def __init__(self, channel_widths, linear_sizes, kernel, pooling, nonlinearity=nn.ReLU()):
        super(ClassifierModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.channel_widths = channel_widths
        self.linear_sizes = linear_sizes
        self.kernel = kernel
        self.pooling = pooling
        self.nonlinearity = nonlinearity
        
        layers = []
        for i in range(len(channel_widths)-2):
            layers.append(nn.Conv2d(channel_widths[i], channel_widths[i+1],
                                    kernel_size=kernel, padding=2, stride=1, bias=True))
            layers.append(nonlinearity)
        layers.append(nn.Conv2d(channel_widths[-2], channel_widths[-1],
                                    kernel_size=kernel, padding=2, stride=1, bias=True))
        self.backbone = nn.Sequential(*layers)
        self.global_pooling = pooling
        self.pool_size = pooling.output_size[0]*pooling.output_size[1]

        # Defining the fully connected layers
        fc_layers = []
        in_features = channel_widths[-1]*self.pool_size
        for size in linear_sizes:
            fc_layers.append(nn.Linear(in_features, size))
            fc_layers.append(nonlinearity)
            in_features = size
        self.fully_connected = nn.Sequential(*fc_layers)

        self.linear = nn.Linear(in_features, 2)  # score each class to obtain logits\
        self.to(self.device)
        
        # Initialize lists to store performance metrics
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_accuracy = 0
        self.best_model_state_dict = None
        self.epochs_trained = 0
        self.training_parameter_history = []
        self.training_time = 0
        
    # forward pass
    def forward(self, x):
        x = x.to(self.device)
        B = x.size(0)
        features = self.backbone(x)
        pooled_features = self.global_pooling(features)
        pooled_features = pooled_features.view(B, -1)
        fc_output = self.fully_connected(pooled_features)
        logits = self.linear(fc_output)
        return logits
    
    # function to record performance metrics
    def record_metrics(self, train_loss, train_acc, val_loss, val_acc):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
    
    # Get validation stats
    def validate(self, dataloader, criterion):
        val_loss = 0
        val_acc = 0
        # set model to eval mode (again, unnecessary here but good practice)
        self.eval()
        # don't compute gradients since we are not updating the model, saves a lot of computation
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                class_logits = self(images)
                loss = criterion(class_logits, targets)
                val_loss += loss.item()
                val_acc += (class_logits.data.max(1)[1]).eq(targets).sum().item()
        return val_loss, val_acc

    # Main training function
    def train_model(self, all_data, training_indices, validation_indices, config, verbose=True, printouts=20):
        # unpack configuration parameters
        lr = config['lr'] # learning rate
        n_epochs = config['n_epochs'] # number of passes (epochs) through the training data
        batch_size = config['batch_size']
        self.training_parameter_history.append({'Epochs Scheduled': n_epochs, 'Learning Rate': lr, 'Batch Size': batch_size, 'Epochs Completed': 0, 
                                                'Training Indices': len(training_indices), 'Validation Indices': len(validation_indices), 'Training Time (s)': 0})
        if printouts > n_epochs: printouts = n_epochs
        previous_epochs = self.epochs_trained

        # set up optimizer and loss function
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # set up dataloaders
        train_sampler = torch.utils.data.SubsetRandomSampler(training_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
        trainloader = torch.utils.data.DataLoader(all_data, batch_size=batch_size, sampler=train_sampler)
        valloader = torch.utils.data.DataLoader(all_data, batch_size=batch_size, sampler=val_sampler)
        
        try:
            for n in range(n_epochs):
                # set model to training mode (unnecessary for this model, but good practice)
                self.train()
                epoch_loss = 0
                epoch_acc = 0
                epoch_start = time.time()
                for images, targets in trainloader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    optimizer.zero_grad() # zero out gradients
                    class_logits = self(images)
                    loss = criterion(class_logits, targets)
                    loss.backward() # backpropagate to compute gradients
                    optimizer.step() # update parameters using stochastic gradient descent
                    # update epoch statistics
                    epoch_loss += loss.item() # batch loss
                    epoch_acc += (class_logits.data.max(1)[1]).eq(targets).sum().item() # number of correct predictions
                    
                # validation
                epoch_loss /= len(trainloader)
                epoch_acc /= len(training_indices)
                val_loss, val_acc = self.validate(valloader, criterion)
                val_loss /= len(valloader)
                val_acc /= len(validation_indices)
                
                # log epoch information
                self.record_metrics(epoch_loss, epoch_acc, val_loss, val_acc)
                
                # save best model's state dict, if necessary
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_model_state_dict = copy.deepcopy(self.state_dict())
                
                if verbose and (n+1) % (int(n_epochs/printouts)) == 0:
                    print('Epoch {}/{}: (Train) Loss = {:.4e}, Acc = {:.4f}, (Val) Loss = {:.4e}, Acc = {:.4f}'.format(
                        n + 1 + previous_epochs,
                        n_epochs + previous_epochs,
                        epoch_loss,
                        epoch_acc,
                        val_loss,
                        val_acc))
                self.epochs_trained += 1
                self.training_parameter_history[-1]['Epochs Completed'] += 1
                epoch_time = time.time() - epoch_start
                self.training_time += epoch_time
                self.training_parameter_history[-1]['Training Time (s)'] += epoch_time
        except KeyboardInterrupt:
            print("Training interrupted. Stopping after completing {} epochs of {} planned.".format(self.epochs_trained-previous_epochs, n_epochs))
        return

    # function to plot training performance metrics
    def plot_model_results(self):
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.semilogy(self.train_losses, color='royalblue')
        plt.xlabel('Epoch')
        plt.title('Training loss')
        plt.grid(True)
        plt.subplot(222)
        plt.plot(self.train_accs, color='darkorange')
        plt.xlabel('Epoch')
        plt.title('Training accuracy')
        plt.grid(True)
        plt.subplot(223)
        plt.plot(self.val_losses, color='royalblue')
        plt.xlabel('Epoch')
        plt.title('Validation loss')
        plt.grid(True)
        plt.subplot(224)
        plt.plot(self.val_accs, color='darkorange')
        plt.xlabel('Epoch')
        plt.title('Validation accuracy')
        plt.grid(True)
        plt.show()

    # Function to update and retrieve total model training time
    def get_training_time(self):
        self.training_time = 0
        for dict in self.training_parameter_history:
            self.training_time += dict['Training Time (s)']
        seconds = self.training_time % 60
        minutes = ((self.training_time-seconds) / 60) % 60
        hours = (((self.training_time-seconds) / 60) - minutes) / 60
        print(f"Model trained for: {hours} hrs, {minutes} mins, {seconds} s")
        return self.training_time
    
    # Test different confidence treshold results on accuracy
    def test_with_thresholds(self, model, dataset, thresholds=np.arange(0.5, 1, 0.01)):
        # test model on withheld test data
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
        model.eval()
        
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                class_probs = torch.nn.functional.softmax(model(images), dim=1)
                all_probs.extend(class_probs.tolist())
                all_targets.extend(targets.tolist())

        results = []
        for threshold in thresholds:
            correct_predictions = 0
            total_predictions = 0
            rejected_predictions = 0

            for i in range(len(all_probs)):
                max_prob = max(all_probs[i])
                predicted_class = all_probs[i].index(max_prob)
                if max_prob >= threshold:
                    total_predictions += 1
                    if predicted_class == all_targets[i]:
                        correct_predictions += 1
                else:
                    rejected_predictions += 1

            test_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
            rejection_ratio = rejected_predictions / len(all_targets)
            results.append((threshold, test_acc, rejection_ratio))
        
        return results

    # Plot results from 'test_with_thresholds'
    def plot_confidence_thresholding(self, datasets, thresholds=np.arange(0.5, 1, 0.01), use_best_model=False, colors=['orange', 'blue', 'green', 'red', 'black']):
        if use_best_model:
            model_state_dict = self.best_model_state_dict
        else:
            model_state_dict = self.state_dict()
        # load the state dict into a new instance of the model for testing
        model = ClassifierModel(self.channel_widths, self.linear_sizes, self.kernel, self.pooling, self.nonlinearity)
        model.load_state_dict(model_state_dict)
        model = model.to(self.device)

        plt.figure()
        for i in range(len(datasets)):
            dataset = datasets[i]
            results = np.array(self.test_with_thresholds(model, dataset[1], thresholds))
            plt.plot(results[:, 0], results[:, 1], color=colors[i], label = dataset[0] + ': Accuracy')
            plt.plot(results[:, 0], results[:, 2], color=colors[i], linestyle='--', label = dataset[0] + ': Rejection Ratio')
        plt.legend()
        ax = plt.gca()
        ax.set_yticks(np.arange(0, 1, 0.05), minor=True)
        ax.set_yticks(np.arange(0, 1.1, 0.1), minor=False)
        ax.grid(True, which='major')
        ax.grid(True, which='minor', ls='--')
        plt.ylabel("Accuracy/Rejection Ratios")
        plt.xlabel("Confidence Threshold")
        plt.title("Model Accuracy with Confidence Thresholding")
        plt.show()

    # Plots a histogram of the confidence levels for all predictions
    def confidence_histogram(self, dataset):
        # Get the model's predictions on the dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
        self.eval()

        confidences = []

        with torch.no_grad():
            for images, targets in dataloader:
                class_probs = torch.nn.functional.softmax(self(images), dim=1)

                # High confidence for class A (label 0) will be close to 1
                # and high confidence for class B (label 1) will be close to 0
                confidence_score = 1 - class_probs[:, 1]

                confidences.extend(confidence_score.tolist())

        # Plot the histogram
        plt.hist(confidences, bins=20, range=(0, 1), alpha=0.7)
        plt.title("Confidence Histogram")
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.show()

    # Plots accuracy on each class with separate columns for correct/incorrect/rejected
    def plot_classification_results(self, dataset, confidence_threshold=0.5):

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
        self.eval()

        correct_counts = {0: 0, 1: 0}
        incorrect_counts = {0: 0, 1: 0} 
        rejected_counts = {0: 0, 1: 0}

        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)

                for label in [0, 1]:
                    is_label = targets == label

                    confident_indices = softmax_outputs[:, label] > confidence_threshold
                    
                    correct_indices = predicted[is_label & confident_indices] == label
                    incorrect_indices = predicted[is_label & confident_indices] != label
                    rejected_indices = ~confident_indices & is_label
                    
                    correct_counts[label] += correct_indices.sum().item()
                    incorrect_counts[label] += incorrect_indices.sum().item()
                    rejected_counts[label] += rejected_indices.sum().item()

        labels = dataset.class_names
        
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots()
        
        correct_bars = ax.bar(x - width, [correct_counts[0], correct_counts[1]], width, label='Correct')
        incorrect_bars = ax.bar(x, [incorrect_counts[0], incorrect_counts[1]], width, label='Incorrect') 
        rejected_bars = ax.bar(x + width, [rejected_counts[0], rejected_counts[1]], width, label='Rejected')

        ax.set_xlabel('Classes')
        ax.set_ylabel('Counts')
        ax.set_title('Classification Results')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.show()

    # Save everything about the model so it can be reloaded later
    def save_model(self, PATH):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_accuracy': self.best_val_accuracy,
            'epochs_trained': self.epochs_trained,
            'training_parameter_history': self.training_parameter_history,
            'training_time': self.training_time,
            'channel_widths': self.channel_widths,
            'linear_sizes': self.linear_sizes,
            'kernel': self.kernel,
            'pooling': self.pooling,
            'nonlinearity': type(self.nonlinearity),  # save the type of nonlinearity
            'best_model_state_dict': self.best_model_state_dict,
        }
        torch.save(checkpoint, PATH)

    # Reload everything in the model from a file
    @classmethod
    def load_model(cls, PATH):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(PATH, map_location=torch.device(device))
        model = cls(
            channel_widths=checkpoint['channel_widths'],
            linear_sizes=checkpoint['linear_sizes'],
            kernel=checkpoint['kernel'],
            pooling=checkpoint['pooling'],
            nonlinearity=checkpoint['nonlinearity'](),  # instantiate the nonlinearity
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train_losses = checkpoint['train_losses']
        model.train_accs = checkpoint['train_accs']
        model.val_losses = checkpoint['val_losses']
        model.val_accs = checkpoint['val_accs']
        model.best_val_accuracy = checkpoint['best_val_accuracy']
        model.epochs_trained = checkpoint['epochs_trained']
        model.training_parameter_history = checkpoint['training_parameter_history']
        model.training_time = checkpoint['training_time']
        model.best_model_state_dict = checkpoint['best_model_state_dict']

        model.to(model.device)
        return model
