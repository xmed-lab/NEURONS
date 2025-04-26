from PIL import Image
from torchvision import transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--exp", type=str, default='',
)
parser.add_argument(
    "--subj", type=int, default=1, choices=[1, 2, 3],
    help="Validate on which subject?",
)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"



processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", cache_dir='./pretrained_weights')
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", cache_dir='./pretrained_weights', torch_dtype=torch.float16
)
model.to(device)

images = torch.load(f'EXP/exp_{args.exp}/subj_{args.subj}/frames_generated/video_subj0{args.subj}_all_recons.pt', map_location='cpu')
print(images.shape)
all_predcaptions = []
for i in range(images.shape[0]):
    print(i)
    x = images[i]
    # print(f"\033[92m xx {x.shape} \033[0m")

    x = transforms.ToPILImage()(x)

    # print(f"\033[92m {np.array(x).shape} \033[0m")

    inputs = processor(images=x, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    all_predcaptions = np.hstack((all_predcaptions, generated_text))
    print(generated_text)

torch.save(all_predcaptions, f'EXP/exp_{args.exp}/subj_{args.subj}/frames_generated/pred_test_caption.pt')

print(f"\033[92m Saved to EXP/exp_{args.exp}/subj_{args.subj}/frames_generated/pred_test_caption.pt \033[0m")

