import math
import torch
import numpy as np


class featureExtractor:
    """
    Extracts feature maps/activate maps from various model

    References:
    (iterate model.named_children() instead of children())- https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904 
    Assignment3 Gatech
    """
    def get_activation_map(self,x_in, model,model_type="swin"):
        """
        add forward hook to the last output of swin transformer sequential block
        """
        
        # get last feature map from model.pretrained
        feature_map_mod = self.get_module_chidren(model=model,model_type=model_type)
        
        self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.
        def activation_hook(a, b, activation):
            self.activation_value = activation
        
        feature_map_mod.register_forward_hook(activation_hook)
        
        # forward
        score = model(x_in)
        self.score = score
        
    def get_module_chidren(self,model,model_type="swin"):
        """
        Helper method to get last featuremap module depending on the architecture
        """
        
        # index depending on model:
        if model_type == "swin":
            last_feature_index = 3
        elif model_type == "resnet34":
            last_feature_index = 7
        
        
        for i,param in enumerate(model.pretrained.children()):
            if i == last_feature_index:
                feature_map_mod = param
        
        return feature_map_mod


class CudaCKA(object):
    """
    https://github.com/jayroxis/CKA-similarity/blob/main/CKA.py

    """
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)