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

class AddGaussianNoise(object):
    """
    https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class normalize(object):
    def __init__(self,mode="norm"):
        self.mode = mode

    def __call__(self, tensor):
        return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

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
                normalize(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), # imgnet requirements 
            ]
        )
        else:
            transformations = T.Compose([
                T.Resize([224,224]),# imgnet needs at least 224
                T.ConvertImageDtype(torch.float),
                normalize(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), # imgnet requirements 
                ]
            )

    if method == "advance":
        if train:
            transformations = T.Compose(
            [
                T.Resize([224,224]),# imgnet needs at least 224
                T.RandomApply(torch.nn.ModuleList([AddGaussianNoise(mean=0,std=0.3)]),p=0.5),
                T.RandomHorizontalFlip(), 
                T.RandomVerticalFlip(),
                T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.RandomApply(torch.nn.ModuleList([T.RandomResizedCrop(size=(224,224),scale=(0.8,1),ratio = (1.7, 2.3))]),p=0.33),
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

    
    if method == "kaggle":
        raise NotImplementedError
    

    return transformations