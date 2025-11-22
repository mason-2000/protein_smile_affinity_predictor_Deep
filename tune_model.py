#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import sys


def save_model(model, mean_test_loss, save_dir="model/"):
    """
    Save the trained model weights and the final test RMSE score.
    """

    print(f"[INFO] saving tuned model in {save_dir}")

    # Save the model state dict to the path provided by Snakemake (sys.argv[2])
    torch.save(model.state_dict(), sys.argv[2])

    # Write the test RMSE to a txt file (sys.argv[3])
    with open(sys.argv[3], "w") as log:
        log.write(str(mean_test_loss))


def test_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set and compute the final RMSE.
    """

    model.eval()
    Test_loss = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move batch to device
            inputs, labels = inputs.to(device), labels.unsqueeze(1).to(device)

            # Forward pass
            labels_pred = model(inputs)

            # Compute RMSE
            Test_loss.append(torch.sqrt(criterion(labels_pred, labels)).item())

    # Mean RMSE over the entire test set
    mean_test_loss = sum(Test_loss) / len(Test_loss)
    print(f"[INFO] Test RMSE: {mean_test_loss:.4f}")

    # Save model + test RMSE
    save_model(model, mean_test_loss)
          

def tune_model(model, train_loader, val_loader, test_loader,
               device, epochs=100, lr=10**(-3),
               best_val_loss=10**6, best_state=None):
    """
    Train the MLP model using RMSE loss, track validation performance,
    and restore the best model before final testing.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()   # Base loss — RMSE computed from it manually

    model.train()
    for epoch in range(epochs):

        mean_train_loss = []

        # ---- Training phase ----
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).to(device)

            # Forward
            labels_pred = model(inputs)

            # RMSE
            Train_loss = torch.sqrt(criterion(labels_pred, labels))
            mean_train_loss.append(Train_loss)

            # Backpropagation
            optimizer.zero_grad()
            Train_loss.backward()
            optimizer.step()

        # ---- Validation phase ----
        model.eval()
        with torch.no_grad():

            Val_loss = []
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.unsqueeze(1).to(device)

                # Forward pass
                labels_pred = model(inputs)

                # RMSE
                Val_loss.append(torch.sqrt(criterion(labels_pred, labels)).item())
                
        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} - Train RMSE: {sum(mean_train_loss)/len(mean_train_loss):.4f}\n"
            f"VAL RMSE: {sum(Val_loss)/len(Val_loss):.4f}"
        )

        # Save best model state based on validation performance
        if sum(Val_loss)/len(Val_loss) < best_val_loss:
            best_val_loss = sum(Val_loss)/len(Val_loss)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.train()

    print("[INFO] tarining phase completed")
    print("-------------------------------------------------------------------")

    # Load best model before final testing
    model.load_state_dict(best_state)

    # Final test evaluation
    test_model(model, test_loader, criterion, device)


def init_datasets(batch_size=64, dataset_path=sys.argv[1]):
    """
    Load the merged embedding dataset and create train/val/test splits.
    """

    # Load protein+drug embedding file (merged_embeddings.pkl)
    dataset = Emb_dataset(dataset_path)

    # Create a list of all sample indices
    indices = list(range(len(dataset)))

    # 70% train, 30% eval (val+test)
    train_idx, eval_idx = train_test_split(indices, test_size=0.3, shuffle=True)

    # Split eval set into 10% val, 20% test (approx)
    val_idx, test_idx = train_test_split(eval_idx, test_size=0.66, shuffle=True)

    # Create Subset wrappers
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    # DataLoaders for batching
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_ds, train_loader, val_ds, val_loader, test_ds, test_loader


class AffinityPredictor(nn.Module):
    """
    A feed-forward neural network for affinity prediction.
    Input dim is (protein_emb_dim + drug_emb_dim).
    """

    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


class Emb_dataset(Dataset):
    """
    Dataset that loads one merged embedding sample at a time
    (protein mean-pooled vector + drug mean-pooled vector).
    """

    def __init__(self, dataset_path=None):
        # Read merged embeddings file
        self.dataset = pd.read_pickle(dataset_path)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load protein & drug embedding vectors
        p_emb = torch.tensor(self.dataset.loc[idx, 'protein_matrix'], dtype=torch.float32)
        s_emb = torch.tensor(self.dataset.loc[idx, 'drugs_matrix'], dtype=torch.float32)

        # Affinity label
        label = self.dataset.loc[idx, 'affinity']

        # Concatenate into a single vector (input for MLP)
        merged_emb = torch.cat((p_emb, s_emb))

        label = torch.tensor(label, dtype=torch.float32)

        return merged_emb, label
    

if __name__ == '__main__':

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] working on {device}")
    print("-------------------------------------------------------------------")
    print("[INFO] tarining phase started")

    # Load dataset + create train/val/test splits
    train_ds, train_loader, val_ds, val_loader, test_ds, test_loader = init_datasets()

    # Infer input dimensionality from first sample
    sample, _ = train_ds[0]
    input_dim = sample.shape[0]

    # Initialize model
    model = AffinityPredictor(input_dim).to(device)
    
    # Train and evaluate
    tune_model(model, train_loader, val_loader, test_loader, device)



