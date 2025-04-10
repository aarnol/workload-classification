import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import scipy.io
import pandas as pd
from scipy.stats import zscore
import math
sampling_rate = 3.8147
def get_onset_index(timing, sampling_rate):
    
    timing = timing / sampling_rate
    return int(timing)
class FNIRSDataset(Dataset):
    def __init__(self, subject_ids, condition='nback', type='HbR', transform=None):
        """
        Args:
            subject_ids (list): List of subject IDs to load data for.
            condition (str): Experimental condition ('nback' or other).
            type (str): fNIRS signal type ('HbO', 'HbR', 'HbT').
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.data = []
        for subject_id in subject_ids:
            self.data.extend(self.load_fnirs_subject(subject_id, condition, type))
        # Pad the data so both dimensions are divisible by 4
        for i in range(len(self.data)):
            sample = self.data[i]['roiTimeseries']
            pad_height = (4 - sample.shape[0] % 4) % 4
            pad_width = (4 - sample.shape[1] % 4) % 4
            self.data[i]['roiTimeseries'] = np.pad(sample, ((0, pad_height), (0, pad_width)), mode='constant')
        self.transform = transform
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]['roiTimeseries']
        label = self.data[idx]['pheno']['label']
        
        if self.transform:
            sample = self.transform(sample)
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
    def load_fnirs_subject(self, subject_id, condition='nback', type='HbR'):
        type_dict = {'HbO': 0, 'HbR': 1, 'HbT': 2}
        target_folder = f"data"
        data_path = os.path.join(target_folder, 'fNIRS_HCP_SubjSpecific 1.mat')
        data = scipy.io.loadmat(data_path)['Data_fNIRS'][subject_id - 1][0]
        
        if condition == 'nback':
            data = data[:, :3]
        else:
            data = data[:, 4:]
        
        data = data[:, type_dict[type]]
        formatted_data = []
        labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]
        conditions = [4, 1, 2, 4, 1, 3, 2, 3, 4, 1, 2, 4, 1, 3, 2, 3]
        
        for i, block in enumerate(data):
            f_data = {
                'roiTimeseries': block,
                'pheno': {
                    'subjectId': f'S{subject_id}',
                    'encoding': None,
                    'label': labels[i],
                    'condition': conditions[i],
                    'modality': 'fNIRS'
                }
            }
            formatted_data.append(f_data)
        
        return formatted_data

# Example usage
def get_fnirs_dataloader(subject_ids, condition='nback', type='HbR', batch_size=32, shuffle=True, num_workers=2):
    dataset = FNIRSDataset(subject_ids, condition, type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class ONRData(Dataset):
    def __init__(self, label_dtype = 'float', downsampled = False, transform=None, feature_extraction=False):
        self.participants = [
            'P07', 'P10', 'P11', 'P13', 'P14', 'P16', 'P17', 'P18', 'P22', 'P23', 
            'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 
            'P34', 'P35', 'P36'
            ]
        self.label_dtype = label_dtype
        self.downsampled = downsampled
        timings = []
        for subject in self.participants:
            path = os.path.join("data", "lsl", subject+"_lsl.tri")
            timing = pd.read_csv(path, sep=";", header=None, names=['time', 'onset', 'code'])
            timing['time'] = pd.to_datetime(timing['time']).dt.strftime('%H:%M:%S')
            for i in [2,3,4,6]: #remove the seventh trial
                #find the sixth and seventh instance of the code
                seventh = timing[timing['code'] == int("1"+str(i))].index[-1]
                sixth = timing[timing['code'] == int("1"+str(i))].index[-2]
                #remove all events in the middle
                timing = timing.drop(timing.index[sixth+1:seventh+1])
                


            timings.append(timing)
        self.timings = timings
        
        data = {}
        if self.downsampled:
            folder=  'data_csvs_downsampled'
        else:
            folder = 'data_csvs'
        for subject in self.participants:
            path = os.path.join("data", folder, subject+".csv")
            fnir = pd.read_csv(path, sep=",", header=0)
            fnir = fnir.drop(columns=['Time'])
            
            

            data[subject] = fnir
         
        
        
        
        
        
        workload = pd.read_csv("data/load/workload.csv")
        workload['timestamp'] = workload['timestamp'].str.replace('.', ':')
        workload_dict = {}
        for i in range(len(self.participants)):
            workload_dict[self.participants[i]] = workload[workload['participant_number'] == int(self.participants[i][1:])].iloc[:, :]
        for i in range(len(self.participants)):
            matched_times = timings[i][timings[i]['time'].isin(workload_dict[self.participants[i]]['timestamp'])]
            workload_dict[self.participants[i]] = workload_dict[self.participants[i]].merge(matched_times, left_on='timestamp', right_on='time', how='left')
        
        #parameters 
        if self.downsampled:
            window_size = 4
        else:
            window_size = math.floor(4 * 3.8147)
        #collating data and labels
        
       
        X = []
        y = []
        groups = []
        channel_structure = pd.read_csv("data/channel_structure.csv", header=None).to_numpy()
       
        #iterate through the subjects
        for sub in self.participants:
            num_samples = 0
            sub_metadata = workload_dict[sub]
            sub_data = data[sub].to_numpy()
            
            
            for row in sub_metadata.itertuples():
                onset = row.onset
                label = row.load_label

                #Get the corresponding fnirs sample 
                if(np.isnan(onset)):
                    continue
                if(self.downsampled):
                    index = get_onset_index(onset, sampling_rate)
                else:
                    index = int(onset)
               
                
                #Calculate the window
                fnirs_sample = sub_data[index - window_size + 1 : index + 1, :]  # Extract window (time, channels)
                
                
                
                
                print("sample shape:" + str(fnirs_sample.shape))
                fnirs_sample_structure = np.zeros((9,8,window_size))
                
                for i in range(9):
                    for j in range(8):
                        for k in range(window_size):
                            
                            channel = int(channel_structure[i,j] -1)
                            
                            
                            fnirs_sample_structure[i,j,k] = fnirs_sample[k][channel]
                fnirs_sample  = fnirs_sample_structure
                if feature_extraction:
                    # The shape is now (9, 8, window_size) - need to calculate features differently
                    # Calculate features across the time dimension (axis=2)
                    mean = np.mean(fnirs_sample, axis=2)  # (9, 8)
                    variance = np.var(fnirs_sample, axis=2) + 1e-8  # (9, 8)
                    skewness = np.zeros_like(mean)
                    kurtosis = np.zeros_like(mean)
                    max_val = np.max(fnirs_sample, axis=2)  # (9, 8)
                    min_val = np.min(fnirs_sample, axis=2)  # (9, 8)
                    range_val = max_val - min_val  # (9, 8)
                    
                    # Calculate higher order statistics 
                    for i in range(9):
                        for j in range(8):
                            # Calculate skewness
                            diff = fnirs_sample[i, j, :] - mean[i, j]
                            skewness[i, j] = np.mean(diff**3) / (variance[i, j] ** 1.5)
                            # Calculate kurtosis
                            kurtosis[i, j] = np.mean(diff**4) / (variance[i, j] ** 2)
                    
                    # Flatten all features to create a feature vector
                    mean_flat = mean.flatten()
                    variance_flat = variance.flatten()
                    skewness_flat = skewness.flatten()
                    kurtosis_flat = kurtosis.flatten()
                    max_flat = max_val.flatten()
                    range_flat = range_val.flatten()
                    
                    # Concatenate all features
                    fnirs_sample = np.concatenate([
                        mean_flat, variance_flat, skewness_flat, 
                        kurtosis_flat, max_flat, range_flat
                    ])
                    
                    print("sample shape after feature extraction:" + str(fnirs_sample.shape))
                
                if np.isnan(fnirs_sample).any():
                    continue
                    
                    
                X.append(fnirs_sample)
                
                
                y.append(label)
                groups.append(sub)
            
            
        X = np.array(X)
        print(np.unique(y))
        #convert labels to integers
        mapping = {'optimal': 0, 'underload': 2, 'overload': 1}
        y = [mapping[label] for label in y]
        #check counts of each label
        counts = np.bincount(y)
        print("Counts of each label:")
        for i, count in enumerate(counts):
            print(f"Label {i}: {count}")

        #remove the labels that are underload
        X = [X[i] for i in range(len(X)) if y[i] != 2]
        groups = [groups[i] for i in range(len(groups)) if y[i] != 2]
        y = [y for y in y if y != 2]

        y = np.array(y)
        
        #try for P10
        # X = [X[i] for i in range(len(X)) if groups[i] == 'P10']
        # y = [y[i] for i in range(len(y)) if groups[i] == 'P10']
        # groups = [groups[i] for i in range(len(groups)) if groups[i] == 'P10']
        print(np.array(X).shape)
        print(np.array(y).shape)
      
        
        self.X = X
        
        self.y = y
        self.groups = groups
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        
        # pad_height = (4 - sample.shape[0] % 4) % 4 
        # pad_width = (4 - sample.shape[1] % 4) % 4 
        sample = torch.permute(sample, (2, 0, 1))


       

        

        # sample = torch.nn.functional.pad(sample, (0, pad_width, 0, pad_height), mode='constant', value=0)
        
       
        
        
        
        if self.label_dtype == 'int':
            return sample.to(torch.float32), torch.tensor(label, dtype=torch.int64)
        else:
            return sample.to(torch.float32), torch.tensor(label, dtype=torch.float32)


if __name__ == '__main__':
    # Test the dataset
    dataset = ONRData()
    # print(len(dataset))
    # print(dataset[0][0].shape)
   
    # print(dataset[0][0].shape)
    # print(dataset[0][0].dtype)
    # print(dataset[0][1].dtype)
    # print(dataset[0][0].max())
    # print(dataset[0][0].min())
    # print(dataset[0][0].mean())
    # print(dataset[0][0].std())