import pandas as pd
import numpy as np
import random
import string
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

n = 3000
df = pd.DataFrame(dict(
    A=np.random.uniform(-0.5, 2, size=n).round(2),
    B=np.random.uniform(-4, 4, size=n).round(2),
    C=0.25,
    Desc = get_random_string(random.randint(25,50))
))

df.to_csv('items.csv',header = 'abcd')

