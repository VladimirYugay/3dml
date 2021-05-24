from pathlib import Path

import torch

from e2.data.shapenet import ShapeNetPoints
from e2.model.pointnet import PointNetClassification


def train(model, trainloader, valloader, device, config):

    loss_criterion = torch.nn.CrossEntropyLoss()
    loss_criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # TODO Implement the training loop. It looks very much the same as in the previous exercise part, except that you are now using points instead of voxel grids
    
    best_accuracy = 0.
    for epoch in range(config['max_epochs']):
        # keep track of running average of train loss for printing
        train_loss_running = 0.

        for i, batch in enumerate(trainloader):
            ShapeNetPoints.move_batch_to_device(batch, device)
            
            optimizer.zero_grad()

            prediction = model(batch['points'])

            loss_total = loss_criterion(prediction, batch['label'])  
                
            loss_total.backward()
            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(trainloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_loss_running = 0.

            # validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()

                loss_total_val = 0
                total, correct = 0, 0
                # forward pass and evaluation for entire validation set
                for batch_val in valloader:
                    ShapeNetPoints.move_batch_to_device(batch_val, device)

                    with torch.no_grad():
                        prediction = model(batch_val['points'])

                    # TODO: Get predicted labels from scores
                    _, predicted_label = torch.max(prediction, dim=1)
                    

                    # TODO: keep track of total / correct / loss_total_val
                    total += predicted_label.shape[0]
                    correct += (predicted_label == batch_val['label']).sum().item()
                    loss_total_val += loss_criterion(prediction, batch_val['label']).item()

                accuracy = 100 * correct / total

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.3f}, val_accuracy: {accuracy:.3f}%')

                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'e2/runs/{config["experiment_name"]}/model_best.ckpt')
                    best_accuracy = accuracy

                # set model back to train
                model.train()


def main(config):
    """
    Function for training PointNet on ShapeNet
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "e2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in e2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = ShapeNetPoints('train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = ShapeNetPoints('train' if not config['is_overfit'] else 'overfit')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    # Instantiate model
    model = PointNetClassification(ShapeNetPoints.num_classes)

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'e2/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)
