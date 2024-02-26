import os
import torch
import argparse
import torchvision


from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler, 
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler, 
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from transformers import T5EncoderModel, T5Tokenizer

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from download import find_model
from pipeline_videogen import VideoGenPipeline
from models import get_models
from utils import save_video_grid
import imageio

def downloadmode_fromHuggingFace():
    from huggingface_hub import snapshot_download
    cache_dir = './'
    snapshot_download(repo_id="maxin-cn/Latte", local_dir=cache_dir)
    return cache_dir

def main( ckpt='', pretrained_model_path='', text_prompt=[], run_time=0, model = 'LatteT2V',
         save_img_path='', video_length = 16, image_size=[512, 512], beta_start = 0.0001, 
         beta_end = 0.02, beta_schedule='linear', variance_type='learned_range', guidance_scale=7.5, 
         sample_method='PNDM', num_sampling_steps=50, enable_temporal_attentions=True,
         enable_vae_temporal_decoder=False, use_compile=False, use_fp16 = True, seed=''):
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transformer_model = get_models(model,pretrained_model_path, video_length, latent_size=None, num_classes=None, 
               num_frames=None, learn_sigma= None, extras = None).to(device, dtype=torch.float16)
    state_dict = find_model(ckpt)
    transformer_model.load_state_dict(state_dict)
    
    if enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if sample_method == 'DDIM':
        scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=beta_start, 
                                                  beta_end=beta_end, 
                                                  beta_schedule=beta_schedule,
                                                  variance_type=variance_type)
    elif sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_path, 
                                                        subfolder="scheduler",
                                                        beta_start=beta_start, 
                                                        beta_end=beta_end, 
                                                        beta_schedule=beta_schedule,
                                                        variance_type=variance_type)
    elif sample_method == 'DDPM':
        scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=beta_start, 
                                                  beta_end=beta_end, 
                                                  beta_schedule=beta_schedule,
                                                  variance_type=variance_type)
    elif sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=beta_start, 
                                                  beta_end=beta_end, 
                                                  beta_schedule=beta_schedule,
                                                  variance_type=variance_type)
    elif sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=beta_start, 
                                                  beta_end=beta_end, 
                                                  beta_schedule=beta_schedule,
                                                  variance_type=variance_type)
    elif sample_method == 'PNDM':
        scheduler = PNDMScheduler.from_pretrained(pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=beta_start, 
                                                  beta_end=beta_end, 
                                                  beta_schedule=beta_schedule,
                                                  variance_type=variance_type)
    elif sample_method == 'HeunDiscrete':
        scheduler = HeunDiscreteScheduler.from_pretrained(pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=beta_start, 
                                                  beta_end=beta_end, 
                                                  beta_schedule=beta_schedule,
                                                  variance_type=variance_type)
    elif sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=beta_start, 
                                                  beta_end=beta_end, 
                                                  beta_schedule=beta_schedule,
                                                  variance_type=variance_type)
    elif sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler.from_pretrained(pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=beta_start, 
                                                  beta_end=beta_end, 
                                                  beta_schedule=beta_schedule,
                                                  variance_type=variance_type)
    elif sample_method == 'KDPM2AncestralDiscrete':
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(pretrained_model_path, 
                                                  subfolder="scheduler",
                                                  beta_start=beta_start, 
                                                  beta_end=beta_end, 
                                                  beta_schedule=beta_schedule,
                                                  variance_type=variance_type)


    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                 text_encoder=text_encoder, 
                                 tokenizer=tokenizer, 
                                 scheduler=scheduler, 
                                 transformer=transformer_model).to(device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    video_grids = []
    for prompt in text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        videos = videogen_pipeline(prompt, 
                                video_length=video_length, 
                                height=image_size[0], 
                                width=image_size[1], 
                                num_inference_steps=num_sampling_steps,
                                guidance_scale=guidance_scale,
                                enable_temporal_attentions=enable_temporal_attentions,
                                num_images_per_prompt=1,
                                mask_feature=True,
                                enable_vae_temporal_decoder=enable_vae_temporal_decoder
                                ).video
        try:
            imageio.mimwrite(save_img_path + prompt.replace(' ', '_') + '_%04d' % run_time + 'webv-imageio.mp4', videos[0], fps=8, quality=9) # highest quality is 10, lowest is 0
        except:
            print('Error when saving {}'.format(prompt))
        video_grids.append(videos)
    video_grids = torch.cat(video_grids, dim=0)

    video_grids = save_video_grid(video_grids)

    # torchvision.io.write_video(save_img_path + '_%04d' % run_time + '-.mp4', video_grids, fps=6)
    imageio.mimwrite(save_img_path + '_%04d' % run_time + '-.mp4', video_grids, fps=8, quality=5)
    print('save path {}'.format(save_img_path))

    # save_videos_grid(video, f"./{prompt}.gif")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, default="./configs/wbv10m_train.yaml")
    # args = parser.parse_args()

    # path:
    cache_dir = downloadmode_fromHuggingFace()
    ckpt = os.path.join(cache_dir, 't2v.pt')
    save_img_path = './sample_output/t2v'
    pretrained_model_path = os.path.join(cache_dir, 't2v_required_models')

    text_prompt= ['A dog in astronaut suit and sunglasses floating in space.',]
    main(ckpt=ckpt, pretrained_model_path=pretrained_model_path, text_prompt=text_prompt, save_img_path=save_img_path, run_time=0, model = 'LatteT2V',
         video_length = 16, image_size=[512, 512], beta_start = 0.0001, 
         beta_end = 0.02, beta_schedule='linear', variance_type='learned_range', guidance_scale=7.5, 
         sample_method='PNDM', num_sampling_steps=50, enable_temporal_attentions=True,
         enable_vae_temporal_decoder=False, use_compile=False, use_fp16 = True, seed='')
    # main(OmegaConf.load(args.config))

