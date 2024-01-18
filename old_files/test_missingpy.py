import numpy as np
import pandas as pd
import logging
import sys
# missingpy and sklearn compatibility
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base # noqa
from missingpy import MissForest

# supress warnings
import warnings
warnings.filterwarnings("ignore")

random_seed = 42
np.random.seed(random_seed)
test = np.random.rand(10,10)
test[0,0] = np.nan
test[3,3] = np.nan
print(test)

imputer = MissForest(n_jobs=-1, verbose=0, random_state=random_seed)
imputer2 = MissForest(n_jobs=-1, verbose=0, random_state=random_seed)


out = imputer.fit_transform(test)
out1 = imputer2.fit_transform(test)

print("Test: ",(out==out1).all(),"\n")

out2 = imputer.fit_transform(test)
print("Test: ",(out==out2).all(),"\n")



print("1 core")

imputer = MissForest(n_jobs=1, verbose=0, random_state=random_seed)
imputer2 = MissForest(n_jobs=1, verbose=0, random_state=random_seed)

out = imputer.fit_transform(test)
out1 = imputer2.fit_transform(test)

print("Test: ",(out==out1).all(),"\n")

out2 = imputer.fit_transform(test)
print("Test: ",(out==out2).all(),"\n")