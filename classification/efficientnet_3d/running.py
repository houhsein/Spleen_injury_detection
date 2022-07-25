import torch
from torch.utils.tensorboard import SummaryWriter


def adjust_learning_rate(optimizer, epoch, init_lr, decay_rate=.5 ,lr_decay_epoch=40):
    ''' Sets the learning rate to initial LR decayed by e^(-0.1*epochs)'''
    lr = init_lr * (decay_rate ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        #param_group['lr'] =  param_group['lr'] * math.exp(-decay_rate*epoch)
        param_group['lr'] = lr
        #lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    #print('LR is set to {}'.format(param_group['lr']))
    return optimizer , lr

def train(model, device, data_num, epochs, optimizer, loss_function, train_loader, valid_loader, early_stop, init_lr, decay_rate, lr_decay_epoch, check_path):
    # Let ini config file can be writted
    global best_metric
    global best_metric_epoch
    #val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    trigger_times = 0

    #epoch_loss_values = list()
    
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        # LR decay 
        optimizer , LR = adjust_learning_rate(optimizer, epoch, init_lr, decay_rate, lr_decay_epoch)
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = data_num // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("desnet_train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        #epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        #if (epoch + 1) % val_interval == 0:
            # Early stopping & save best weights by using validation
        metric = validation(model, valid_loader, device)
        # checkpoint setting
        if metric > best_metric:
            # reset trigger_times
            trigger_times = 0
            best_metric = metric
            best_metric_epoch = epoch + 1

            torch.save(model.state_dict(), f"{check_path}/{best_metric}.pth")
            print('trigger times:', trigger_times)
            print("saved new best metric model")
        else:
            trigger_times += 1
            print('trigger times:', trigger_times)
        print(
            "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )
        writer.add_scalar("desnet_val_accuracy", metric, epoch + 1)

        # early stop 
        if trigger_times >= early_stop:
            print('Early stopping!\nStart to test process.')
            print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
            return model
        
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    return model


def validation(model, val_loader, device):
    #metric_values = list()
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
            val_outputs = model(val_images)
            value = torch.eq(val_outputs.argmax(dim=1), val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
        metric = num_correct / metric_count
        #metric_values.append(metric)
    return metric