import torch
import os


def train(dataloader, model, loss_fn, optimizer, currentEpoch, writer):
    model.train()
    print('Current Epoch:', currentEpoch)
    correct = 0
    size = len(dataloader.dataset)
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        y = y.cuda()
        # Compute prediction and loss
        anchor, pos, test1 = model(X)
        loss, pred, disa, dist, minus, y = loss_fn(anchor, pos, test1, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_equal = torch.eq(pred, y).sum().item()
        correct += num_equal
        train_loss += loss.item()
        if batch % 352 == 0:
            print('disa', disa)
            print('dist', dist)
            print('minus', minus)
            print('y', y)
            current = (batch + 1) * 16
            losscurrent = train_loss / current
            print(f"loss: {losscurrent:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    print(f"Train Error: \n Accuracy: {(100 * correct):>0.01f}%")
    writer.add_scalar(tag="loss/train",
                      scalar_value=loss,
                      global_step=currentEpoch
                      )
    writer.add_scalar(tag="accuracy/train",
                      scalar_value=correct * 100,
                      global_step=currentEpoch
                      )
    # path = os.path.join(r'E:\file\Code\deepLearning\signature', currentEpoch + '.pth')
    # torch.save(model.state_dict(), path)


def test(dataloader, model, loss_fn, currentEpoch, writer):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = y.cuda()
            anchor, pos, test1 = model(X)
            loss, pred, disa, dist, minus, y = loss_fn(anchor, pos, test1, y)
            test_loss += loss.item()
            correct += torch.eq(pred, y).sum().item()

    test_loss /= num_batches
    correct /= size

    writer.add_scalar(tag="loss/val",
                      scalar_value=test_loss,
                      global_step=currentEpoch
                      )
    writer.add_scalar(tag="accuracy/val",
                      scalar_value=correct * 100,
                      global_step=currentEpoch
                      )
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.01f}%, Avg loss: {test_loss:>8f} \n")
