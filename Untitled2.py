#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load all necessary packages
import sys
sys.path.insert(1, "../")  

import numpy as np
np.random.seed(0)

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric, DatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer

from IPython.display import Markdown, display

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import json
from collections import OrderedDict


# In[3]:


dataset_orig = GermanDataset(protected_attribute_names=['age'],           # this dataset also contains protected
                                                                          # attribute for "sex" which we do not
                                                                          # consider in this evaluation
                             privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
                             features_to_drop=['personal_status', 'sex']) # ignore sex-related attributes

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]


# In[4]:


print("Original one hot encoded german dataset shape: ",dataset_orig.features.shape)
print("Train dataset shape: ", dataset_orig_train.features.shape)
print("Test dataset shape: ", dataset_orig_test.features.shape)


# In[5]:


df, dict_df = dataset_orig.convert_to_dataframe()


# In[6]:


print("Shape: ", df.shape)
print(df.columns)
df.head(5)


# In[7]:


df['age'].value_counts().plot(kind='bar')
plt.xlabel("Age (0 = under 25, 1 = over 25)")
plt.ylabel("Frequency")


# In[8]:


print("Key: ", dataset_orig.metadata['label_maps'])
df['credit'].value_counts().plot(kind='bar')
plt.xlabel("Credit (1 = Good Credit, 2 = Bad Credit)")
plt.ylabel("Frequency")


# In[9]:


metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Original training dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


# In[10]:


print("Original training dataset")
print("Disparate Impact = %f" % metric_orig_train.disparate_impact())


# In[11]:


print("Original training dataset")
print("Disparate Impact = %f" % metric_orig_train.disparate_impact())


# In[12]:


text_expl = MetricTextExplainer(metric_orig_train)
json_expl = MetricJSONExplainer(metric_orig_train)


# In[13]:


print(text_expl.mean_difference())


# In[14]:


print(text_expl.disparate_impact())


# In[15]:


def format_json(json_str):
    return json.dumps(json.loads(json_str, object_pairs_hook=OrderedDict), indent=2)


# In[16]:


print(format_json(json_expl.mean_difference()))


# In[17]:


RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(dataset_orig_train)


# In[18]:


dataset_transf_train.instance_weights


# In[19]:


len(dataset_transf_train.instance_weights)


# In[21]:


metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
print("Transformed training dataset")
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())


# In[22]:


print("Transformed training dataset")
print("Disparate Impact = %f" % metric_transf_train.disparate_impact())


# In[ ]:




