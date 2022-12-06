import sys, time
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

if len(sys.argv) < 2:
  print("Please enter a prompt in quotes")
  quit()

prompt = sys.argv[1]
filename = sys.argv[2] if len(sys.argv) >2 else prompt.replace(" ","_")
filename = filename + "-" + str(int(time.time())) + ".png"

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
device = torch.device("mps")
print("device %s" % device)
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, safety_checker=None).to(device)
pipe.enable_attention_slicing()


image = pipe(prompt, height=768, width=768, num_inference_steps=25).images[0]

image.save("./images/" + filename)
print("wrote %s to images dir" % filename)
