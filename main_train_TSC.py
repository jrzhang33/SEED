import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
from datautil.getdataloader_single import get_act_dataloader #follow the official Library: robustlearn

from torch.utils.data import TensorDataset, ConcatDataset, DataLoader


var_size = {
    'emg': {
        'in_size': 8,
        'ker_size': 9,
        'fc_size': 32*44
    }
}


class ActNetwork(nn.Module): #same as DIVERSITY
    def __init__(self, taskname):
        super(ActNetwork, self).__init__()
        self.taskname = taskname
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=var_size[taskname]['in_size'], out_channels=16, kernel_size=(1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, var_size[taskname]['ker_size'])),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5)
        )
        self.in_features = var_size[taskname]['fc_size']
        self.fc = nn.Linear(self.in_features, var_size[taskname]['num_classes'])  

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        try:
            x = x.view(-1, self.in_features)
        except:
            x = x.contiguous().view(-1, self.in_features)
        x = self.fc(x) 
        return x

def load_data(file_path, batch_size=64):
    data = torch.load(file_path)
    
    x_train = data['x']
    y_train = data['y']

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def train_model(model, train_loader, test_loader, device, epochs=120, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    print("Training Completed.")

    evaluate_model(model, test_loader, device)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--task_id', default=1, type=int)
    parser.add_argument('--run_id', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--data_path', default='./datasets', type=str)
    parser.add_argument('--dataset', default='emg', type=str)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    testuser = {
        'seed': args.run_id,
        'name': f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}",
        'dataset': args.dataset,
        'newdata': os.path.join(os.getcwd(), 'intermediate_results', f"{args.dataset}_task_{args.task_id}_seed_{args.run_id}-newdata.pt"),
    }

    # Load original data
    train_loader_orig, _, _, _ = get_act_dataloader(args)
    orig_data, orig_labels = [], []

    for xb, yb, *_ in train_loader_orig:
        orig_data.append(xb)
        orig_labels.append(yb)
    orig_data = torch.cat(orig_data, dim=0)
    orig_labels = torch.cat(orig_labels, dim=0)

    orig_dataset = TensorDataset(orig_data, orig_labels)

    # Load generated data
    new_data = torch.load(testuser['newdata'])
    gen_dataset = TensorDataset(new_data['x'].float(), new_data['y'].long())

    # Combine datasets
    combined_dataset = ConcatDataset([orig_dataset, gen_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

    # Load test set
    test_loader, _, _, _ = get_act_dataloader(args)

    # Train model
    model = ActNetwork(args.dataset)
    train_model(model, train_loader, test_loader, device, epochs=120, lr=0.001)
