#%%
import numpy as np
import matplotlib.pyplot as plt
# %%
import matplotlib.image as mpimg
plt.figure(dpi=200, figsize=(5,5))

i  =  12
img_cls = mpimg.imread('fig/mode_{}_cls_losses.png'.format(i))
img_dis  = mpimg.imread('fig/mode_{}_data_distribution.png'.format(i))
img_reg  = mpimg.imread('fig/mode_{}_reg_losses.png'.format(i))
plt.imshow(img_dis)
plt.imshow(img_cls)
plt.imshow(img_reg)
plt.axis('off')
# %%
img_cls = mpimg.imread('fig/mode_{}_cls_losses.png'.format(i))
plt.imshow(img_cls)
# %%
class Preprocessor:
    def __init__(self):
        self.params = None
    def fit_transform(self, data_features, data_carbs, data_vegs):
        self.params = [np.std(data_features, axis=0, keepdims=True), np.mean(data_features, axis=0, keepdims=True), 
                       np.std(data_carbs, axis=0, keepdims=True), np.mean(data_carbs, axis=0, keepdims=True)]
        return self.transform(data_features, data_carbs, data_vegs)
    def transform(self, data_features, data_carbs, data_vegs):
        if self.params is None:
            raise ValueError('Not fit yet')
        return (data_features-self.params[1])/self.params[0], (data_carbs-self.params[3])/self.params[2], data_vegs-1
    def inverse(self, data_features_trans, data_carbs_trans, data_vegs_trans):
        return data_features_trans*self.params[0]+self.params[1], data_carbs_trans*self.params[2]+self.params[3], data_vegs_trans+1