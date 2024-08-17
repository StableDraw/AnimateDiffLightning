import torch
import av
import io
import numpy
from os.path import isdir, isfile
from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pytorch_lightning import seed_everything

device = "cuda"
dtype = torch.float16

def image_array_to_binary_video(video, fps, video_codec = "libx264", is_image_list = False, options = None, audio_array = None, audio_fps = None, audio_codec = None, audio_options = None):
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        video (Tensor[T, H, W, C]): tensor containing the individual frames or binary image list,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        is_image_list (bool): is input image list or 4d image tensor
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """

    if is_image_list == False:
        video_array = torch.as_tensor(video, dtype = torch.uint8).numpy()
    else:
        video_array = []
        for frame in video:
            video_array.append(numpy.asarray(Image.open(io.BytesIO(frame)).convert("RGB")))
        video_array = numpy.array(video_array)

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = numpy.round(fps)

    binary_video = io.BytesIO()

    with av.open(binary_video, mode = "w", format = "mp4") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate = audio_fps)
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = container.streams.audio[0].format.name

            format_dtype = numpy.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)

            frame = av.AudioFrame.from_ndarray(audio_array, format = audio_sample_fmt, layout=audio_layout)

            frame.sample_rate = audio_fps

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format = "rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

    return binary_video.getbuffer()


def text_to_video(prompt, opt):

    model_name = f"animatediff_lightning_{opt["step"]}step_diffusers.safetensors"
    if not isfile("weights/animate/" + model_name):
        hf_hub_download(repo = "ByteDance/AnimateDiff-Lightning", ckpt = model_name, local_dir = "weights/animate", local_dir_use_symlinks = False) #Загрузка весов

    pretrained_model_name_or_path = "weights/base/" + opt["base"]
    if not isdir(pretrained_model_name_or_path):
        pipe = StableDiffusionPipeline.from_single_file(pretrained_model_name_or_path + ".safetensors")
        pipe.save_pretrained(pretrained_model_name_or_path)

    adapter = MotionAdapter().to(device, dtype)

    adapter.load_state_dict(load_file(f"weights/animate/animatediff_lightning_{opt["step"]}step_diffusers.safetensors", device = device))

    seed_everything(opt["seed"])

    pipe = AnimateDiffPipeline.from_pretrained(f"weights/base/{opt["base"]}", use_safetensors=True, motion_adapter = adapter, torch_dtype = dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing = "trailing", beta_schedule = "linear")

    output = pipe(prompt = prompt, num_frames = opt["num_frames"], height = opt["height"], width = opt["width"], num_inference_steps = opt["num_inference_steps"], guidance_scale = opt["guidance_scale"], negative_prompt = opt["negative_prompt"], num_videos_per_prompt = opt["num_videos_per_prompt"], eta = opt["eta"], generator = opt["generator"], latents = opt["latents"], prompt_embeds = opt["prompt_embeds"], negative_prompt_embeds = opt["negative_prompt_embeds"], ip_adapter_image = opt["ip_adapter_image"], ip_adapter_image_embeds = opt["ip_adapter_image_embeds"], output_type = opt["output_type"], return_dict = opt["return_dict"], cross_attention_kwargs = opt["cross_attention_kwargs"], clip_skip = opt["clip_skip"], callback_on_step_end = opt["callback_on_step_end"], callback_on_step_end_tensor_inputs = opt["callback_on_step_end_tensor_inputs"])

    torch.cuda.empty_cache() #Очищаем видеопамять

    return output



def external_text_to_video(prompt):
    '''
    Функция для вызова извне, без необходимости настройки параметров
    Принимает текстовое описание и то, возвращать видео или список кадров в виде изображений
    '''

    args = {
        "step": 8, # Options: [1,2,4,8]
        "repo": "ByteDance/AnimateDiff-Lightning",
        "base": "mistoonAnime_v30", #Базовые модели, доступно: (аниме модели: "mistoonAnime_v30", "imp_v10"), "epiCRealism"
        "fps": 10, #Количество кадров в секунду при экспорте в видео
        "seed": 43, #Сид

        "height": 320, #The height in pixels of the generated video.
        "width": 512, #The width in pixels of the generated video.
        "num_frames": 16, #The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds amounts to 2 seconds of video.
        "num_inference_steps": 50, #The number of denoising steps. More denoising steps usually lead to a higher quality videos at the expense of slower inference.
        "guidance_scale": 1.0, #A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        "negative_prompt": None, #The prompt or prompts to guide what to not include in image generation. If not defined, you need to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        "num_videos_per_prompt": 1, #Количество видео для одного описания
        "eta": 0.0, #Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        "generator": None, #A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
        "latents": None, #Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`. Latents should be of shape `(batch_size, num_channel, num_frames, height, width)`.
        "prompt_embeds": None, #Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.
        "negative_prompt_embeds": None, #Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
        "ip_adapter_image": None, #Optional image input to work with IP Adapters.
        "ip_adapter_image_embeds": None, #Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
        "output_type": "bytes", #The output format of the generated video. Choose between `torch.FloatTensor`, `PIL.Image` or `np.array` or "bytes".
        "return_dict": True, #Whether or not to return a [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] instead of a plain tuple.
        "cross_attention_kwargs": None, #A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        "clip_skip": None, #Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings.
        "callback_on_step_end": None, #A function that calls at the end of each denoising steps during the inference. The function is called  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        "callback_on_step_end_tensor_inputs": ["latents"] #The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeine class.
    } 

    binary_list = text_to_video(prompt = prompt, opt = args)

    binary_video = image_array_to_binary_video(video = binary_list[0], fps = args["fps"], video_codec = "libx264", is_image_list = True)

    return binary_video



if __name__ == "__main__":
    '''
    args = {
        "step": 8, # Options: [1,2,4,8]
        "repo": "ByteDance/AnimateDiff-Lightning",
        "base": "mistoonAnime_v30", #Базовые модели, доступно: (аниме модели: "mistoonAnime_v30", "imp_v10"), "epiCRealism"
        "fps": 10, #Количество кадров в секунду при экспорте в видео
        "seed": 43, #Сид

        "height": 320, #The height in pixels of the generated video.
        "width": 512, #The width in pixels of the generated video.
        "num_frames": 16, #The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds amounts to 2 seconds of video.
        "num_inference_steps": 50, #The number of denoising steps. More denoising steps usually lead to a higher quality videos at the expense of slower inference.
        "guidance_scale": 1.0, #A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        "negative_prompt": None, #The prompt or prompts to guide what to not include in image generation. If not defined, you need to pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        "num_videos_per_prompt": 1, #Количество видео для одного описания
        "eta": 0.0, #Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        "generator": None, #A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation deterministic.
        "latents": None, #Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video generation. Can be used to tweak the same generation with different prompts. If not provided, a latents tensor is generated by sampling using the supplied random `generator`. Latents should be of shape `(batch_size, num_channel, num_frames, height, width)`.
        "prompt_embeds": None, #Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, text embeddings are generated from the `prompt` input argument.
        "negative_prompt_embeds": None, #Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
        "ip_adapter_image": None, #Optional image input to work with IP Adapters.
        "ip_adapter_image_embeds": None, #Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
        "output_type": "bytes", #The output format of the generated video. Choose between `torch.FloatTensor`, `PIL.Image` or `np.array` or "bytes".
        "return_dict": True, #Whether or not to return a [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] instead of a plain tuple.
        "cross_attention_kwargs": None, #A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        "clip_skip": None, #Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that the output of the pre-final layer will be used for computing the prompt embeddings.
        "callback_on_step_end": None, #A function that calls at the end of each denoising steps during the inference. The function is called  with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        "callback_on_step_end_tensor_inputs": ["latents"] #The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the `._callback_tensor_inputs` attribute of your pipeine class.
    } 

    prompt = "A girl smiling"

    binary_list = text_to_video(prompt = prompt, opt = args)

    return_video = True #Возвращать видео или список изображений кадров

    if return_video == True:
        binary_video = image_array_to_binary_video(video = binary_list[0], fps = args["fps"], video_codec = "libx264", is_image_list = True)

        # Write BytesIO from RAM to file, for testing
        with open("output.mp4", "wb") as f:
            f.write(binary_video)

    for i, img in enumerate(binary_list[0]):
        Image.open(io.BytesIO(img)).save(f"result/{i}.png")
    '''

    prompt = "A girl smiling"

    output = external_text_to_video(prompt = prompt)

    with open("output.mp4", "wb") as f:
            f.write(output)