# 使用 PyTorch Geometric 的圖神經網絡

## 概述

- [`dataset_featurizer.py:`](#dataset_featurizerpy) 建構符合 PyTorch Geometric 格式的 GNN 資料集。
- [`main.py:`](#mainpy) 建構 GAT 模型，並使用 Optuna 進行調參。

## 相關套件

- PyTorch
- PyTorch Geometric
- Optuna
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- tqdm

# dataset_featurizer.py

- 使用網絡資料與節點特徵資料，建構符合 PyTorch Geometric 格式的資料集。

### 分析流程

1. **原始資料**
   ```python
   import pandas as pd
   network_df = pd.read_csv(networkPath)
   feature_df = pd.read_csv(featurePath)
   ```

2. **資料篩選**
   ```python
    # Filter data based on specified conditions
    mask = "based on your specified conditions"
    subset = network_df[mask]   

    # Group by specified conditions
    data_list = []
    grouped = subset.groupby(["based on your specified conditions"])
    print("total group:",len(grouped.groups))
   ```

3. **建構 PyTorch Geometric 格式的資料**
      
   簡略版:

   ```python
   from torch_geometric.data import Data

   data = Data(x=torch.tensor(x, dtype=torch.float), 
               edge_index=torch.tensor(edge_index, dtype=torch.long), 
               y=torch.tensor(y, dtype=torch.long))
   data_list.append(data)
   ```

4. **save dataset**
   ```python
   # Collate the list of Data objects into a single dataset
   data, slices = self.collate(data_list)
   
   # Save the processed dataset
   torch.save((data, slices), self.processed_paths[0])
   ```

# main.py

- 定義、訓練和評估 GNN 模型，並使用 Optuna 進行超參數的調參。

### 分析流程

1. **讀取資料集**
   ```python
   import torch
   dataset = torch.load(dataset_path)
   ```

2. **定義 GNN 模型**

   舉例 node classification，需根據不同 task 進行修改:

   ```python

    class GAT(torch.nn.Module):
        def __init__(self, num_hidden, num_features, num_classes, heads = 8):
            super().__init__()
            self.conv1 = GATConv(num_features, num_hidden, heads)
            self.conv2 = GATConv(heads*num_hidden, num_classes, heads)
    
        def forward(self, params, x, edge_index):
            """
            Forward pass for the GNN model.
            
            Args:
                params (dict): Dictionary of hyperparameters.
                x (torch.Tensor): Node features.
                edge_index (torch.Tensor): Edge indices.
            
            Returns:
                torch.Tensor: Output predictions.
            """
            x = F.dropout(x, p=0.3)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3)
            x = self.conv2(x, edge_index)
    
            return x
   ```

4. **訓練與評估模型**
   
   模型訓練包含 early stopping 的機制，可以防止過擬合。評估指標包括 confusion matrix、F1-score、accuracy、precision、recall、ROC、AUC
   
   簡略版:

   ```python
   from torch_geometric.loader import DataLoader

   def train(model, optimizer, loader):
       model.train()
       total_loss = 0
       for data in loader:
           optimizer.zero_grad()
           out = model(data.x, data.edge_index, data.batch)
           loss = F.nll_loss(out, data.y)
           loss.backward()
           optimizer.step()
           total_loss += loss.item()
       return total_loss / len(loader)

   def test(model, loader):
       model.eval()
       correct = 0
       for data in loader:
           out = model(data.x, data.edge_index, data.batch)
           pred = out.argmax(dim=1)
           correct += (pred == data.y).sum().item()
       return correct / len(loader.dataset)
   ```

6. **使用 Optuna 進行超參數的調參**
   
   簡略版:

   ```python
   import optuna

   def objective(trial):
       lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
       model = GNNModel(dataset.num_node_features, dataset.num_classes)
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)

       train_loader = DataLoader(dataset[:800], batch_size=32, shuffle=True)
       test_loader = DataLoader(dataset[800:], batch_size=32, shuffle=False)

       for epoch in range(50):
           train_loss = train(model, optimizer, train_loader)
           test_acc = test(model, test_loader)

       return test_acc

   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100)
   ```

8. **呈現 Optuna 調參後最好的超參數**
   
   簡略版:

   ```python
   best_trial = study.best_trial
   print(f"Best trial: {best_trial.value}")
   print(f"Best hyperparameters: {best_trial.params}")
   ```

## 參考資料

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [Optuna](https://optuna.org/)

