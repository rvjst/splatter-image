import argparse
import json
import os
import sys
import tqdm
import numpy as np
from omegaconf import OmegaConf

from huggingface_hub import hf_hub_download

import lpips as lpips_lib

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from datasets.dataset_factory import get_dataset
from utils.loss_utils import ssim as ssim_fn

import imageio

from utils.app_utils import (
    remove_background, 
    resize_foreground, 
    set_white_background,
    resize_to_128,
    to_tensor,
    get_source_camera_v2w_rmo_and_quats,
    get_target_cameras,
    export_to_obj)
# 세 가지 다른 메트릭(PSNR, SSIM, LPIPS)을 계산
class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target):
        lpips = self.lpips_net( image.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - target) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, target).item()
        return psnr, ssim, lpips

@torch.no_grad()
def evaluate_dataset(model, dataloader, device, model_cfg, save_vis=1, out_folder=None
                     ):
    """
    Runs evaluation on the dataset passed in the dataloader. 
    Computes, prints and saves PSNR, SSIM, LPIPS.
    Args:
        save_vis: how many examples will have visualisations saved
    """
    # save_vis 매개변수가 0보다 큰 경우, 결과를 저장할 폴더를 생성
    if save_vis > 0: 
        out_folder = "/home/hamdol/splatter-image" # delete
        os.makedirs(out_folder, exist_ok=True)

    # score 텍스트 파일 생성
    with open("scores.txt", "w+") as f:
        f.write("")

    # background 설정
    bg_color = [1, 1, 1] if model_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # instantiate metricator
    metricator = Metricator(device)

    psnr_all_examples_novel = []
    ssim_all_examples_novel = []
    lpips_all_examples_novel = []

    psnr_all_examples_cond = []
    ssim_all_examples_cond = []
    lpips_all_examples_cond = []

    # 데이터로더에서 각 예제를 반복하여 평가를 수행
    for d_idx, data in enumerate(tqdm.tqdm(dataloader)):
        
        # 각 데이터에 대해 새로운 재구성 및 평가를 위한 빈 리스트들을 초기화
        psnr_all_renders_novel = []
        ssim_all_renders_novel = []
        lpips_all_renders_novel = []
        psnr_all_renders_cond = []
        ssim_all_renders_cond = []
        lpips_all_renders_cond = []

        # 데이터를 모델이 처리할 수 있는 장치로 이동데이터를 모델이 처리할 수 있는 장치로 이동
        data = {k: v.to(device) for k, v in data.items()}

        # 필요에 따라 입력 이미지의 회전 변환(quaternion)을 추출
        rot_transform_quats = data["source_cv2wT_quat"][:, :model_cfg.data.input_images]

        if model_cfg.data.category == "hydrants" or model_cfg.data.category == "teddybears":
            focals_pixels_pred = data["focals_pixels"][:, :model_cfg.data.input_images, ...]
        else:
            focals_pixels_pred = None

        # 입력 이미지를 구성하고, 필요에 따라 원본 거리 정보를 추가
        if model_cfg.data.origin_distances:
            input_images = torch.cat([data["gt_images"][:, :model_cfg.data.input_images, ...],
                                      data["origin_distances"][:, :model_cfg.data.input_images, ...]],
                                      dim=2)
        else:
            input_images = data["gt_images"][:, :model_cfg.data.input_images, ...]

        # 현재 데이터의 예제 ID를 가져와 시각화 결과를 저장할 폴더를 생성
        example_id = dataloader.dataset.get_example_id(d_idx)
        if d_idx < save_vis:

            out_example_gt = os.path.join(out_folder, "{}_".format(d_idx) + example_id + "_gt")
            out_example = os.path.join(out_folder, "{}_".format(d_idx) + example_id)

            os.makedirs(out_example_gt, exist_ok=True)
            os.makedirs(out_example, exist_ok=True)

        # 모델을 사용하여 입력 이미지를 재구성
        # batch has length 1, the first image is conditioning
        reconstruction = model(input_images,
                               data["view_to_world_transforms"][:, :model_cfg.data.input_images, ...],
                               rot_transform_quats,
                               focals_pixels_pred)

        # 각 이미지에 대해 루프를 수행하며 다음을 수행
        loop_renders = []
        t_to_128 = torchvision.transforms.Resize(128, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        for r_idx in range( data["gt_images"].shape[1]):
            # 예측된 초점 픽셀 값이 있는 경우, 해당 값 가져옴
            if "focals_pixels" in data.keys():
                focals_pixels_render = data["focals_pixels"][0, r_idx]
            else:
                focals_pixels_render = None
            # 이미지를 예측하고 시각화
            image = render_predicted({k: v[0].contiguous() for k, v in reconstruction.items()},
                                     data["world_view_transforms"][0, r_idx],
                                     data["full_proj_transforms"][0, r_idx], 
                                     data["camera_centers"][0, r_idx],
                                     background,
                                     model_cfg,
                                     focals_pixels=focals_pixels_render)["render"]
            # 시각화 결과를 저장할 경우 해당 이미지를 저장
            image = t_to_128(image)
            loop_renders.append(torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        
            if d_idx < save_vis:
                # vis_image_preds(reconstruction, out_example)
                torchvision.utils.save_image(image, os.path.join(out_example, '{0:05d}'.format(r_idx) + ".png"))
                torchvision.utils.save_image(data["gt_images"][0, r_idx, ...], os.path.join(out_example_gt, '{0:05d}'.format(r_idx) + ".png"))
            
            # 메트릭을 계산하고 이를 적절한 리스트에 추가, 이때, 입력 이미지의 메트릭은 "cond" 리스트에, 새로운 이미지의 메트릭은 "novel" 리스트에 추가
            # exclude non-foreground images from metric computation
            if not torch.all(data["gt_images"][0, r_idx, ...] == 0):
                psnr, ssim, lpips = metricator.compute_metrics(image, data["gt_images"][0, r_idx, ...])
                if r_idx < model_cfg.data.input_images:
                    psnr_all_renders_cond.append(psnr)
                    ssim_all_renders_cond.append(ssim)
                    lpips_all_renders_cond.append(lpips)
                else:
                    psnr_all_renders_novel.append(psnr)
                    ssim_all_renders_novel.append(ssim)
                    lpips_all_renders_novel.append(lpips)

        # 루프 중간에 생성된 이미지 시퀀스를 .mp4 파일로 저장
        loop_out_path = os.path.join(os.path.dirname("./mesh_one.ply"), "loop.mp4")
        imageio.mimsave(loop_out_path, loop_renders, fps=25)
        # export reconstruction to ply
        # 함수를 사용하여 재구성 결과를 .ply 파일로 내보냄
        export_to_obj(reconstruction, "./mesh_one.ply")

        # 각 입력 이미지의 메트릭 값을 평균하여 리스트에 추가
        psnr_all_examples_cond.append(sum(psnr_all_renders_cond) / len(psnr_all_renders_cond))
        ssim_all_examples_cond.append(sum(ssim_all_renders_cond) / len(ssim_all_renders_cond))
        lpips_all_examples_cond.append(sum(lpips_all_renders_cond) / len(lpips_all_renders_cond))

        psnr_all_examples_novel.append(sum(psnr_all_renders_novel) / len(psnr_all_renders_novel))
        ssim_all_examples_novel.append(sum(ssim_all_renders_novel) / len(ssim_all_renders_novel))
        lpips_all_examples_novel.append(sum(lpips_all_renders_novel) / len(lpips_all_renders_novel))

        with open("scores.txt", "a+") as f:
            f.write("{}_".format(d_idx) + example_id + \
                    " " + str(psnr_all_examples_novel[-1]) + \
                    " " + str(ssim_all_examples_novel[-1]) + \
                    " " + str(lpips_all_examples_novel[-1]) + "\n")

    # scores를 반환
    # scores = {"PSNR_cond": sum(psnr_all_examples_cond) / len(psnr_all_examples_cond),
    #           "SSIM_cond": sum(ssim_all_examples_cond) / len(ssim_all_examples_cond),
    #           "LPIPS_cond": sum(lpips_all_examples_cond) / len(lpips_all_examples_cond),
    #           "PSNR_novel": sum(psnr_all_examples_novel) / len(psnr_all_examples_novel),
    #           "SSIM_novel": sum(ssim_all_examples_novel) / len(ssim_all_examples_novel),
    #           "LPIPS_novel": sum(lpips_all_examples_novel) / len(lpips_all_examples_novel)}        

    return scores

# 이 함수는 코드애서 사용하지 않음
@torch.no_grad()
def eval_robustness(model, dataloader, device, model_cfg, out_folder=None):
    """
    Evaluates robustness to shift and zoom
    """
    # 결과를 저장할 폴더를 생성
    os.makedirs(out_folder, exist_ok=True)

    # background color 설정, background 텐서로 변환
    bg_color = [1, 1, 1] if model_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 특정 인덱스에 해당하는 데이터를 dataloader에서 가져옴. 데이터를 모델이 처리할 수 있는 장치로 이동
    obj_idx = 98

    data = {k: v.unsqueeze(0) for k, v in dataloader.dataset[obj_idx].items()}
    data = {k: v.to(device) for k, v in data.items()}
    
    rot_transform_quats = data["source_cv2wT_quat"][:, :model_cfg.data.input_images]
    focals_pixels_pred = None

    # 입력 이미지를 gt_images에서 추출. 이 이미지는 나중에 줌 및 이동 변형을 적용 받음.
    input_images = data["gt_images"][:, :model_cfg.data.input_images, ...]
    example_id = dataloader.dataset.get_example_id(obj_idx)

    # 이미지의 크기를 128으로 조정, 리사이즈할 때 사용할 BILINEAR 보간 방법을 지정
    resize_to_128_transform = transforms.Resize(128, 
        interpolation=transforms.InterpolationMode.BILINEAR)

    for test_zoom_idx, crop_size in enumerate([-40, -30, -20, -10, 0, 10, 20, 30, 40]):

        # ================ zoom transforms ===============
        # crop_size가 양수인 경우 이미지를 크롭하고, 음수인 경우 이미지를 패딩
        if crop_size >= 0:
            # crop the source images
            input_images = data["gt_images"][
                        0, :model_cfg.data.input_images,  
                        :, crop_size:model_cfg.data.training_resolution-crop_size, crop_size:model_cfg.data.training_resolution-crop_size]
        elif crop_size < 0:
            # pad only the source images
            padding_transform = transforms.Pad(padding=-crop_size,
                                                fill=1.0)
            input_images = padding_transform(data["gt_images"][0, :model_cfg.data.input_images])

        # 이미지를 크기 128로 리사이즈
        if crop_size != 0:
            input_images = resize_to_128_transform(input_images)
        

        # ================ shift transforms ===============
        # 이미지를 이동
        x_shift = 0
        y_shift = crop_size
        padding_transform = transforms.Pad(padding=(abs(x_shift), abs(y_shift)),
                                                fill=1.0)

        padded_source  = padding_transform(data["gt_images"][0, :model_cfg.data.input_images])
        y_start = abs(y_shift) + y_shift
        x_start = abs(x_shift) + x_shift
        input_images = padded_source[ :, :, 
                                        y_start : model_cfg.data.training_resolution + y_start,
                                        x_start : model_cfg.data.training_resolution + x_start]

        # 이동된 이미지를 패딩한 후, 필요한 부분만 잘라냄
        input_images = input_images.unsqueeze(0)
        
        # 각 변형에 대해 결과를 저장할 폴더를 생성
        out_example_gt = os.path.join(out_folder, "{}_".format(test_zoom_idx) + example_id + "_gt")
        out_example = os.path.join(out_folder, "{}_".format(test_zoom_idx) + example_id)

        os.makedirs(out_example_gt, exist_ok=True)
        os.makedirs(out_example, exist_ok=True)

        # batch has length 1, the first image is conditioning
        # 모델을 사용하여 재구성을 수행
        reconstruction = model(input_images,
                                data["view_to_world_transforms"][:, :model_cfg.data.input_images, ...],
                                rot_transform_quats,
                                focals_pixels_pred)
        for r_idx in range( data["gt_images"].shape[1]):
            if "focals_pixels" in data.keys():
                focals_pixels_render = data["focals_pixels"][0, r_idx]
            else:
                focals_pixels_render = None
            # 예측 이미지를 생성
            image = render_predicted({k: v[0].contiguous() for k, v in reconstruction.items()},
                                        data["world_view_transforms"][0, r_idx],
                                        data["full_proj_transforms"][0, r_idx], 
                                        data["camera_centers"][0, r_idx],
                                        background,
                                        model_cfg,
                                        focals_pixels=focals_pixels_render)["render"]

            #예측 이미지와 원본 이미지를 지정된 폴더에 저장
            torchvision.utils.save_image(image, os.path.join(out_example, '{0:05d}'.format(r_idx) + ".png"))
            torchvision.utils.save_image(data["gt_images"][0, r_idx, ...], os.path.join(out_example_gt, '{0:05d}'.format(r_idx) + ".png"))


@torch.no_grad()
def main(dataset_name, experiment_path, device_idx, split='test', save_vis=0, out_folder=None):
    
    # set device and random seed
    # CUDA 장치를 설정, 해당 장치를 선택
    device = torch.device("cuda:{}".format(device_idx))
    torch.cuda.set_device(device)

    # args.experiment_path가 None인 경우, hf_hub_download 함수를 사용하여 Hugging Face Hub에서 구성 파일과 모델 파일을 다운로드
    if args.experiment_path is None:
        cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                 filename="config_{}.yaml".format(dataset_name))

        if dataset_name in ["gso", "objaverse"]:
            model_name = "latest"
        else:
            model_name = dataset_name
        model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                            filename="model_{}.pth".format(model_name))
    
    # 해당 경로에서 구성 파일과 모델 파일을 로드   
    else:
        cfg_path = os.path.join(experiment_path, "hydra", "config.yaml")
        model_path = os.path.join(experiment_path, "model_latest.pth")
    # load cfg
    # 구성 파일을 로드
    training_cfg = OmegaConf.load(cfg_path)

    # check that training and testing datasets match if not using official models 
    # args.experiment_path가 주어진 경우, 훈련 및 테스트 데이터셋이 일치하는지 확인
    if args.experiment_path is not None:
        if dataset_name == "gso":
            # GSO model must have been trained on objaverse
            assert training_cfg.data.category == "objaverse", "Model-dataset mismatch"
        # 그렇지 않은 경우, 모델과 데이터셋 이름이 일치하는지 확인
        else:
            assert training_cfg.data.category == dataset_name, "Model-dataset mismatch"

    # load model
    # 모델을 인스턴스화
    model = GaussianSplatPredictor(training_cfg)
    # 모델 가중치를 로드
    ckpt_loaded = torch.load(model_path, map_location=device)
    # 모델에 가중치를 적용
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    # 모델을 CUDA 장치로 이동
    model = model.to(device)
    # 평가 모드로 전환
    model.eval()
    print('Loaded model!')

    # override dataset in cfg if testing objaverse model
    if training_cfg.data.category == "objaverse" and split in ["test", "vis"]:
        training_cfg.data.category = "gso"
    # instantiate dataset loader
    # 데이터로더를 생성, 배치 크기는 1로 설정되고, 셔플링 없이 순차적으로 데이터를 로드
    dataset = get_dataset(training_cfg, split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            persistent_workers=True, 
                            pin_memory=True, num_workers=1) # num_workers = 1
    # 모델을 평가. 평가 결과는 scores 변수에 저장
    scores = evaluate_dataset(model, dataloader, device, training_cfg, save_vis=save_vis, out_folder=out_folder)
    # split이 vis가 아닌 경우, 평가 점수를 출력
    # if split != 'vis':
    #     print(scores)

    return scores

# 함수는 모델 평가를 위해 필요한 인자를 명령줄에서 받아옴
def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model')
    # 평가할 데이터셋의 이름을 지정
    parser.add_argument('dataset_name', type=str, help='Dataset to evaluate on', 
                        choices=['objaverse', 'gso', 'cars', 'chairs', 'hydrants', 'teddybears', 'nmr'])
    # 모델이 저장된 상위 폴더의 경로를 지정
    parser.add_argument('--experiment_path', type=str, default=None, help='Path to the parent folder of the model. \
                        If set to None, a pretrained model will be downloaded')
    # 평가할 데이터셋의 분할을 지정
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'vis', 'train'],
                        help='Split to evaluate on (default: test). \
                        Using vis renders loops and does not return scores - to be used for visualisation. \
                        You can also use this to evaluate on the training or validation splits.')
    # 렌더링 결과를 저장할 출력 폴더를 지정
    parser.add_argument('--out_folder', type=str, default='out', help='Output folder to save renders (default: out)')
    # 렌더링 결과를 저장할 예제의 수를 지정
    parser.add_argument('--save_vis', type=int, default=1, help='Number of examples for which to save renders (default: 0)')
    return parser.parse_args()

if __name__ == "__main__":
    
    # 명령줄 인자를 파싱
    args = parse_arguments()

    # 명령줄 인자에서 받아온 값을 변수에 저장하고, 해당 변수 값을 기반으로 콘솔에 출력 메시지를 표시
    dataset_name = args.dataset_name
    print("Evaluating on dataset {}".format(dataset_name))
    experiment_path = args.experiment_path
    if args.experiment_path is None:
        print("Will load a model released with the paper.")
    else:
        print("Loading a local model according to the experiment path")
    split = args.split
    if split == 'vis':
        print("Will not print or save scores. Use a different --split to return scores.")
    out_folder = args.out_folder
    save_vis = args.save_vis
    if save_vis == 0:
        print("Not saving any renders (only computing scores). To save renders use flag --save_vis")

    # 모델을 로드하고 데이터셋을 평가
    scores = main(dataset_name, experiment_path, 0, split=split, save_vis=save_vis, out_folder=out_folder)
    # save scores to json in the experiment folder if appropriate split was used
    # split이 vis가 아닌 경우 평가 점수를 JSON 파일로 저장
    if split != "vis":
        # experiment_path가 제공된 경우 해당 경로에 점수를 저장하고, 그렇지 않으면 데이터셋 이름과 분할 이름을 포함하는 파일 이름으로 점수를 저장
        if experiment_path is not None:
            score_out_path = os.path.join(experiment_path, 
                                   "{}_scores.json".format(split))
        else:
            score_out_path = "{}_{}_scores.json".format(dataset_name, split)
        # json.dump를 사용하여 점수를 파일에 저장
        with open(score_out_path, "w+") as f:
            json.dump(scores, f, indent=4)
            
