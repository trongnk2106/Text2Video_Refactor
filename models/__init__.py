import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .latte import Latte_models
from .latte_img import LatteIMG_models
from .latte_t2v import LatteT2V

from torch.optim.lr_scheduler import LambdaLR


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(model,pretrained_model_path, video_length, latent_size=None, num_classes=None, 
               num_frames=None, learn_sigma= None, extras = None
               ):

    if 'LatteIMG' in model:
        return LatteIMG_models[model](
                input_size=latent_size,
                num_classes=num_classes,
                num_frames=num_frames,
                learn_sigma=learn_sigma,
                extras=extras
            )
    elif 'LatteT2V' in model:
        pretrained_model_path = pretrained_model_path
        return LatteT2V.from_pretrained_2d(pretrained_model_path, subfolder="transformer", video_length=video_length)
    elif 'Latte' in model:
        return Latte_models[model](
                input_size=latent_size,
                num_classes=num_classes,
                num_frames=num_frames,
                learn_sigma=learn_sigma,
                extras=extras
            )
    else:
        raise '{} Model Not Supported!'.format(model)
    