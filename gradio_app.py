import torch
import torchvision
import numpy as np

import os
from omegaconf import OmegaConf
from PIL import Image 

# 
from utils.app_utils import (
    remove_background, 
    resize_foreground, 
    set_white_background,
    resize_to_128,
    to_tensor,
    #  이미지 루프 내에서 사용할 카메라의 시점과 회전 정보를 텐서로 반환
    get_source_camera_v2w_rmo_and_quats,
    # 물체를 둘러싸고 있는 루프를 형성하기 위해 사용될 여러 카메라의 위치와 각 카메라의 투영 행렬을 반환
    get_target_cameras,
    export_to_obj)

import imageio

# 3D 포인트 클라우드를 2D 이미지로 변환하고, 이를 통해 신경망을 사용하여 다양한 예측을 수행
from scene.gaussian_predictor import GaussianSplatPredictor
# 3D 가우시안 파라미터를 포함하는 예측 딕셔너리를 사용하여 장면을 렌더링
# 렌더링된 이미지, 화면 공간 포인트, 가시성 필터, 반지름 등을 포함하는 딕셔너리를 반환
from gaussian_renderer import render_predicted

import gradio as gr

import rembg

from huggingface_hub import hf_hub_download

@torch.no_grad()
def main():

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # 파일 로드, model_cfg 변수에 저장
    model_cfg = OmegaConf.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
                    "gradio_config.yaml"
                    ))
    # huggingface download
    model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-multi-category-v1", 
                                filename="model_latest.pth")

    # 가져온 파일을 가지고 3D 포인트 클라우드를 2D 이미지로 변환 및 예측 수행
    model = GaussianSplatPredictor(model_cfg)

    # 사전 학습된 모델의 체크포인트를 로드하고, 이를 현재 모델에 적용한 후, 모델을 지정된 장치로 이동시키는 기능을 수행
    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model.to(device)

    # ============= image preprocessing =============
    rembg_session = rembg.new_session()

    # input image가 있는지 확인
    def check_input_image(input_image):
        if input_image is None:
            raise gr.Error("No image uploaded!")
    # 입력 이미지 전처리
    def preprocess(input_image, preprocess_background=True, foreground_ratio=0.65):
        # 0.7 seems to be a reasonable foreground ratio
        if preprocess_background:
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = set_white_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = set_white_background(image)
        image = resize_to_128(image)
        return image
    # ply 파일 생성 위치
    ply_out_path = f'./mesh.ply'

    # 주어진 이미지를 모델을 통해 재구성하고, 재구성된 출력을 다양한 형식으로 저장
    def reconstruct_and_export(image):
        """
        Passes image through model, outputs reconstruction in form of a dict of tensors.
        """
        # 이미지 전처리 및 모델 입력 준비
        image = to_tensor(image).to(device)
        view_to_world_source, rot_transform_quats = get_source_camera_v2w_rmo_and_quats()
        view_to_world_source = view_to_world_source.to(device)
        rot_transform_quats = rot_transform_quats.to(device)

        # 모델을 통해 이미지 재구성
        reconstruction_unactivated = model(
            image.unsqueeze(0).unsqueeze(0),
            view_to_world_source,
            rot_transform_quats,
            None,
            activate_output=False)

        # 출력 처리
        reconstruction = {k: v[0].contiguous() for k, v in reconstruction_unactivated.items()}
        reconstruction["scaling"] = model.scaling_activation(reconstruction["scaling"])
        reconstruction["opacity"] = model.opacity_activation(reconstruction["opacity"])

        # render images in a loop
        # 타겟 카메라 설정 및 렌더링 준비
        world_view_transforms, full_proj_transforms, camera_centers = get_target_cameras()
        background = torch.tensor([1, 1, 1] , dtype=torch.float32, device=device)
        loop_renders = []
        t_to_512 = torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        
        example_id = '1'
        d_idx = 1
        out_folder = "/home/hamdol/splatter-image"
        out_example = os.path.join(out_folder, "{}_".format(d_idx) + example_id)  
        os.makedirs(out_example, exist_ok=True) 

        
        # 이미지 렌더링 및 저장
        for r_idx in range( world_view_transforms.shape[0]):
            image = render_predicted(reconstruction,
                                        world_view_transforms[r_idx].to(device),
                                        full_proj_transforms[r_idx].to(device), 
                                        camera_centers[r_idx].to(device),
                                        background,
                                        model_cfg,
                                        focals_pixels=None)["render"]
            image = t_to_512(image)
            torchvision.utils.save_image(image, os.path.join(out_example, '{0:05d}'.format(r_idx) + ".png"))
            loop_renders.append(torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        # 비디오 파일로 저장
        loop_out_path = os.path.join(os.path.dirname(ply_out_path), "loop.mp4")
        imageio.mimsave(loop_out_path, loop_renders, fps=25)
        # export reconstruction to ply
        export_to_obj(reconstruction_unactivated, ply_out_path)

        # 경로 반환
        return ply_out_path, loop_out_path

    css = """
    h1 {
        text-align: center;
        display:block;
    }
    """

    # 주어진 이미지를 전처리하고, 재구성한 후, 결과를 파일로 저장하여 반환
    def run_example(image):
        # 이미지 전처리
        preprocessed = preprocess(image)
        # 이미지 재구성 및 파일 저장
        ply_out_path, loop_out_path = reconstruct_and_export(np.array(preprocessed))
        # 결과 반환
        return preprocessed, ply_out_path, loop_out_path

    # Gradio를 사용하여 Splatter Image 모델의 데모를 구축
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # Splatter Image

            **Splatter Image (CVPR 2024)** [[code](https://github.com/szymanowiczs/splatter-image), [project page](https://szymanowiczs.github.io/splatter-image)] is a fast, super cheap-to-train method for object 3D reconstruction from a single image. 
            The model used in the demo was trained on **Objaverse-LVIS on 2 A6000 GPUs for 3.5 days**.
            Locally, on an NVIDIA V100 GPU, reconstruction (forward pass of the network) can be done at 38FPS and rendering (with Gaussian Splatting) at 588FPS.
            Upload an image of an object or click on one of the provided examples to see how the Splatter Image does.
            The 3D viewer will render a .ply object exported from the 3D Gaussians, which is only an approximation.
            For best results run the demo locally and render locally with Gaussian Splatting - to do so, clone the [main repository](https://github.com/szymanowiczs/splatter-image).
            """
            )
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGBA",
                        sources="upload",
                        type="pil",
                        elem_id="content_image",
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)
                with gr.Row():
                    with gr.Group():
                        preprocess_background = gr.Checkbox(
                            label="Remove Background", value=True
                        )
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")

                with gr.Row(variant="panel"): 
                    gr.Examples(
                        examples=[
                            './demo_examples/01_bigmac.png',
                            './demo_examples/02_hydrant.jpg',
                            './demo_examples/03_spyro.png',
                            './demo_examples/04_lysol.png',
                            './demo_examples/05_pinapple_bottle.png',
                            './demo_examples/06_unsplash_broccoli.png',
                            './demo_examples/07_objaverse_backpack.png',
                            './demo_examples/08_unsplash_chocolatecake.png',
                            './demo_examples/09_realfusion_cherry.png',
                            './demo_examples/10_triposr_teapot.png'
                        ],
                        inputs=[input_image],
                        cache_examples=False,
                        label="Examples",
                        examples_per_page=20,
                    )

            with gr.Column():
                with gr.Row():
                    with gr.Tab("Reconstruction"):
                        with gr.Column():
                            output_video = gr.Video(value=None, width=512, label="Rendered Video", autoplay=True)
                            output_model = gr.Model3D(
                                height=512,
                                label="Output Model",
                                interactive=False
                            )

        gr.Markdown(
            """
            ## Comments:
            1. If you run the demo online, the first example you upload should take about 4.5 seconds (with preprocessing, saving and overhead), the following take about 1.5s.
            2. The 3D viewer shows a .ply mesh extracted from a mix of 3D Gaussians. This is only an approximations and artefacts might show.
            3. Known limitations include:
            - a black dot appearing on the model from some viewpoints
            - see-through parts of objects, especially on the back: this is due to the model performing less well on more complicated shapes
            - back of objects are blurry: this is a model limiation due to it being deterministic
            4. Our model is of comparable quality to state-of-the-art methods, and is **much** cheaper to train and run.

            ## How does it work?

            Splatter Image formulates 3D reconstruction as an image-to-image translation task. It maps the input image to another image, 
            in which every pixel represents one 3D Gaussian and the channels of the output represent parameters of these Gaussians, including their shapes, colours and locations.
            The resulting image thus represents a set of Gaussians (almost like a point cloud) which reconstruct the shape and colour of the object.
            The method is very cheap: the reconstruction amounts to a single forward pass of a neural network with only 2D operators (2D convolutions and attention).
            The rendering is also very fast, due to using Gaussian Splatting.
            Combined, this results in very cheap training and high-quality results.
            For more results see the [project page](https://szymanowiczs.github.io/splatter-image) and the [CVPR article](https://arxiv.org/abs/2312.13150).
            """
        )

        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=preprocess,
            inputs=[input_image, preprocess_background],
            outputs=[processed_image],
        ).success(
            fn=reconstruct_and_export,
            inputs=[processed_image],
            outputs=[output_model, output_video],
        )

    demo.queue(max_size=1)
    demo.launch()

if __name__ == "__main__":
    main()
