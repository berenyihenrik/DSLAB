#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install optuna')


# In[5]:


import optuna


# In[9]:


import joblib

study = joblib.load("drive/MyDrive/Colab Notebooks/ELTE/DSLAB/DSLAB 2/study.pkl")

print(study.best_trial.params)


# # Plotting the study

# Plotting the optimization history of the study.

# In[10]:


optuna.visualization.plot_optimization_history(study)


# Plotting the accuracies for each hyperparameter for each trial.

# In[11]:


optuna.visualization.plot_slice(study)


# Plotting the accuracy surface for the hyperparameters.

# In[16]:


optuna.visualization.plot_contour(study)


# In[15]:


optuna.visualization.plot_contour(study, params=["hidden_dim", "latent_dim"])

