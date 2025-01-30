import scipy.io as sio
import os 

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data","data.mat"))
# Load the .mat file
mat_contents = sio.loadmat(path, simplify_cells=True)
print(mat_contents['structList'][0].keys())

