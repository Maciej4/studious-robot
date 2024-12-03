"""
Example taken from huggingface CLIP documentation and lightly modified.
https://huggingface.co/openai/clip-vit-base-patch32

Cool features:
- Model takes about 50ms per image on CPU and about 5ms on GPU
- Surprisingly accurate at identifying images even in Minecraft
"""
from PIL import Image
import time

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# Uncomment the following line and comment the line above to use the GPU
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map="cuda")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open('./images/2024-11-10_21.06.49.png')

labels = ["a photo of a cat", "a photo of a dog", "a minecraft tree", "a minecraft plains biome", "a minecraft ocean",
          "zombie"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

# Also uncomment the following line to use the GPU
# inputs = inputs.to("cuda")

# Run the model 10 times to get an average runtime
for i in range(10):
    start = time.time_ns()
    outputs = model(**inputs)
    print((time.time_ns() - start) / 1e6, "ms")

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

flat_probs = probs.cpu().detach().numpy().flatten()
sorted_results = sorted(zip(labels, flat_probs), key=lambda x: x[1], reverse=True)

for label, prob in sorted_results:
    print(f"{label}: {prob:.5f}")
