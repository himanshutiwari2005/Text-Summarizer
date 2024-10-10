# Import libraries

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Working with data

data = pd.read_excel("summarizer/Data/news.xlsx")
data.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)

## Splitting the data
short = data['Short']
summary = data['Headline']

# for decoder sequence
summary = summary.apply(lambda x: '<go> ' + x + '<stop>')
print(summary.head())
