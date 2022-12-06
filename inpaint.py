import os, time, uuid
import cv2, numpy, torch
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

model_id = "stabilityai/stable-diffusion-2-inpainting"
WIDTH = 768
HEIGHT = 768
image_file = "./images/a_cowboy_riding_a_triceratops-1670270324.png"

# generate the mask for the image
mask_prompts = ['the cowboy']
cliptime1 = time.time()
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
image = Image.open(image_file)
mask_prompts = ["a cowboy"]
inputs = processor(text=mask_prompts, images=[image] * len(mask_prompts), padding=True, return_tensors="pt")
outputs = model(**inputs)
sigmoided = torch.sigmoid(outputs.logits)
detached = sigmoided.detach().numpy()

# we have a tensor, make a mask image
filename = f"{uuid.uuid4()}.png"
plt.imsave(filename, detached)
img2 = cv2.imread(filename)
gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
(thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
mask = Image.fromarray(numpy.uint8(bw_image)).convert('RGB')
mask = mask.resize((WIDTH, HEIGHT))
mask = ImageOps.invert(mask)
os.remove(filename)
mask.save("mask_"+filename)
cliptime2 = time.time()
print("CLIPseg time: %d" % int(cliptime2 - cliptime1))

# now do the inpainting
original = Image.open(image_file)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, image=original, safety_checker=None).to(device)
pipe.enable_attention_slicing()
prompt = "a dense jungle"
image = pipe(prompt, image=original, mask_image=mask, height=HEIGHT, width=WIDTH, num_inference_steps=25).images[0]

image.save("inpainted_%d.png" % int(time.time()))
