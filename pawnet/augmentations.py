import torchvision.transforms as T
import numpy as np
import torch
import random

class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg

class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg

class AddGaussianNoise(torch.nn.Module):
    """
    https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
    """
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class MinMaxScalerVectorized(torch.nn.Module):
    """
    MinMax Scaler

    Transforms each channel to the range [a, b].
    https://discuss.pytorch.org/t/using-scikit-learns-scalers-for-torchvision/53455/8

    This solves the nan training loss from my naive implementation since i did not 
    account for divide by 0 (if max and min are equal)
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ==========
        feature_range : tuple
            Desired range of transformed data.
        """
        self.__dict__.update(kwargs)

    def __call__(self, tensor):
        # Feature range
        a, b = self.feature_range

        dist = tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
        tensor.mul_(b - a).add_(a)

        return tensor * 255 # added this just to scale back to [0,255]


def get_augmentation(train=True,method="basic"):
    """
    Generates T.Compose augmentations based on method (experiments)
    """
    if method == "basic":
        if train:
            transformations = T.Compose(
            [
                T.Resize([224,224]),# imgnet needs at least 224
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), # imgnet requirements 
            ]
        )
        else:
            transformations = T.Compose([
                T.Resize([224,224]),# imgnet needs at least 224
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), # imgnet requirements 
                ]
            )

    if method == "advance":
        if train:
            transformations = T.Compose(
            [
                T.Resize([224,224]),# imgnet needs at least 224
                # T.RandomApply(torch.nn.ModuleList([AddGaussianNoise(mean=0,std=0.3)]),p=0.5),
                T.RandomHorizontalFlip(), 
                T.RandomVerticalFlip(),
                T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.RandomApply(torch.nn.ModuleList([T.RandomResizedCrop(size=(224,224),scale=(0.8,1))]),p=0.5),
                T.ConvertImageDtype(torch.float),
                # MinMaxScalerVectorized(feature_range = (0.,1.)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), # imgnet requirements 
            ]
        )
        else:
            transformations = T.Compose([
                T.Resize([224,224]),# imgnet needs at least 224
                T.ConvertImageDtype(torch.float),
                # MinMaxScalerVectorized(feature_range = (0.,1.)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), # imgnet requirements 
                ]
            )

    
    if method == "kaggle":
        raise NotImplementedError
    

    return transformations