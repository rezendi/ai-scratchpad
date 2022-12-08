import os, sys, time, uuid
import cv2, numpy, torch
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

if len(sys.argv) < 4:
  print("Please enter an original image, text to identify what to retain, and a prompt with which to replace the rest")
  print("You may additionally optionally enter an output filename and the number of inpainting iterations")
  quit()

# parse the args
image_file = sys.argv[1]
mask_prompts = sys.argv[2].split(",")
prompt = sys.argv[3]
filename = sys.argv[2] if len(sys.argv) >2 else prompt.replace(" ","_")
filename = filename + "-" + str(int(time.time())) + ".png"
output_file = sys.argv[4] if len(sys.argv) >4 else "%s_inpainted" % image_file
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 10

# define our models
model_id = "stabilityai/stable-diffusion-2-inpainting"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# load initial image, size it down to something reasonable if needed
# TODO: ensure both dimensions are multiples of 8, which something downstream requires
original = Image.open(image_file)
while original.width > 1024:
  original = original.resize((original.width // 2, original.height // 2))
print("Immage dimensions %d x %d" % (original.width, original.height))

# Get rid of alpha channel, repurposed from https://stackoverflow.com/a/35859141
if original.mode in ('RGBA', 'LA') or (original.mode == 'P' and 'transparency' in original.info):
  converted = original.convert('RGBA')
  background = Image.new('RGBA', converted.size, (255,255,255))
  alpha_composite = Image.alpha_composite(background, converted)
  alpha_composite_3 = alpha_composite.convert('RGB')
  original = alpha_composite_3

# generate mask tensor from the CLIPseg model
inputs = processor(text=mask_prompts, images=[original] * len(mask_prompts), padding=True, return_tensors="pt")
outputs = model(**inputs)
sigmoided = torch.sigmoid(outputs.logits)
detached = sigmoided.detach().numpy()

# we have a tensor, make a mask image
tempfilename = f"{uuid.uuid4()}.png"
plt.imsave(tempfilename, detached)
img2 = cv2.imread(tempfilename)
gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
(thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
mask = Image.fromarray(numpy.uint8(bw_image)).convert('RGB')
mask = mask.resize((original.width, original.height))
mask = ImageOps.invert(mask)
os.remove(tempfilename)
# uncomment to visually inspect the mask image
# mask.save("mask_"+tempfilename) 

# now do the inpainting
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, safety_checker=None).to(device)
pipe.enable_attention_slicing()

image = pipe(prompt, image=original, mask_image=mask, height=original.height, width=original.width, num_inference_steps=steps).images[0]
image.save("%s_%d.png" % (output_file, int(time.time())))
