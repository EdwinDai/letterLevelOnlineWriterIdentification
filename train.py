import torch
import os


def train(dataloader, model, loss_fn, optimizer, writer, currentEpoch):
    model.train()
    print('Current Epoch:', currentEpoch)
    correct = 0
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        y = y.cuda()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 58 == 0:
            loss, current = loss.item(), (batch + 1) * 64
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
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


def test(dataloader, model, loss_fn, writer, currentEpoch):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

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
