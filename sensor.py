import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix

# ini
np.random.seed(11)

###
#
mu_x1, sigma_x1 = 0, 0.1

#
x2_mu_diff = 0.35

#
d1 = pd.DataFrame({})