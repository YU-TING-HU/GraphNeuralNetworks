from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import pandas as pd 
import torch 

# Paths and parameters
import os
saveDataPath = "Path to save the dataset"
networkPath = "Path to network data"
featurePath = "Path to feature data"
Sex = "your data sex"
AGE = "your data age"
model = "Model_1"

# Set up log file
import contextlib
file_path = os.path.join(saveDataPath, f"NW_S{Sex}A{AGE}{model}.txt")
with open(file_path, "w") as o:
    with contextlib.redirect_stdout(o):

        class myDataset(InMemoryDataset):

            """
            Custom dataset for handling graph data using PyTorch Geometric.

            Args:
                root (str): Root directory where the dataset should be saved.
                transform (callable, optional): A function/transform that takes in an
                    object and returns a transformed version. The data object will be transformed
                    before every access. (default: None)
                pre_transform (callable, optional): A function/transform that takes in
                    an object and returns a transformed version. The data object will be transformed
                    before being saved to disk. (default: None)
            """

            def __init__(self, root, transform=None, pre_transform=None):
                super(myDataset, self).__init__(root, transform, pre_transform)
                # Load the processed data
                self.data, self.slices = torch.load(self.processed_paths[0])
        
            @property
            def raw_file_names(self):
                """
                Return the file names for the raw data.
                """
                return []
            
            @property
            def processed_file_names(self):
                """
                Return the file names for the processed data.
                """
                return [f"Dataset_S{Sex}A{AGE}{model}.dataset"]
        
            def download(self):
                """
                Download raw data if necessary.
                """
                pass
            
            def process(self):
                """
                Process the raw data and save it in a format suitable for PyTorch Geometric.
                """
                # Load network data
                network_df = pd.read_csv(networkPath)
                # Load feature data
                feature_df = pd.read_csv(os.path.join(featurePath, f"NW_S{Sex}A{AGE}_feature.csv"), encoding='Big5', index_col=False)
                
                # Filter data based on specified conditions
                mask = "based on your specified conditions"
                subset = network_df[mask]   

                # Group by specified conditions
                data_list = []
                grouped = subset.groupby(["based on your specified conditions"])
                print("total group:",len(grouped.groups))

                for grouped_df in tqdm(grouped):

                    # label
                    """
                    Process of label value.
                    Return labels variable.
                    """

                    # edge
                    """
                    Process of edge index.
                    Return edge_index variable.
                    """                    

                    # edge feature
                    """
                    Process of edge feature.
                    Return edge_atr variable.
                    """ 

                    # node feature
                    """
                    Process of node feature.
                    Return x variable.
                    """                     

                    # Create a Data object
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_atr, y=labels)
                    data_list.append(data)
                    
                # Collate the list of Data objects into a single dataset
                data, slices = self.collate(data_list)
                # Save the processed dataset
                torch.save((data, slices), self.processed_paths[0])

        # Instantiate the dataset
        dataset = myDataset(root = saveDataPath)