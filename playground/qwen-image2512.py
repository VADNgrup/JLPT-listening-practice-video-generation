from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
    torch_dtype=torch.bfloat16,
    use_auth_token='xxx',
).to('cuda')

# uncomment if you run out of memory
# pipe.enable_model_cpu_offload() 

output = pipe(
    prompt="a kawaii sloth playing the drums",
    negative_prompt="blurry, unfocused",
    num_inference_steps=20,
    true_cfg_scale=4.0,
)

# Save output
image = output.images[0]
image.save('sample.png')