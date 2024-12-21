import pandas as pd
from pymrmre import mrmr
import numpy as np


n = 500
m = 10000
X = pd.DataFrame(np.random.rand(n, m), columns=[f'x{i}' for i in range(m)])
y = pd.DataFrame(np.random.randint(0, 2, size=(n, 1)), columns=['y'])


solutions = mrmr.mrmr_ensemble(features=X,targets=y,solution_length=5,solution_count=1)
print(solutions.iloc[0][0])
