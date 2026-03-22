import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import skimage
from skimage.segmentation import find_boundaries
from PIL import Image
import time

device ='cuda'
sam = sam_model_registry ["vit_h"] (checkpoint ="./sam_vit_h_4b8939.pth")
sam.to( device = device )

mask_generator = SamAutomaticMaskGenerator(sam, crop_nms_thresh=0.5, box_nms_thresh=0.5, pred_iou_thresh=0.96)

def SAMAug(tI , mask_generator):
    masks = mask_generator.generate(tI)
    if len(masks) == 0:
        return
    tI= skimage.img_as_float (tI)

    BoundaryPrior = np.zeros (( tI. shape [0] , tI. shape [1]))
    BoundaryPrior_output = np.zeros ((tI.shape [0] , tI. shape [1]))
    
    Objects_first_few =  np.zeros (( tI. shape [0] , tI. shape [1]))
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    idx=1
    for ann in sorted_anns:        
        if ann['area'] < 50:
            continue
        if idx==51:
            break
        m = ann['segmentation']
        color_mask = idx
        print(color_mask)
        Objects_first_few[m] = color_mask
        idx=idx+1

    for maskindex in range(len(masks)):
        thismask =masks[ maskindex ][ 'segmentation']
        mask_=np.zeros (( thismask.shape ))
        mask_[np.where( thismask == True)]=1
        BoundaryPrior = BoundaryPrior + find_boundaries (mask_ ,mode='thick')

    BoundaryPrior [np.where( BoundaryPrior >0) ]=1
    BoundaryPrior_index=np.where(BoundaryPrior >0)
    Objects_first_few[BoundaryPrior_index]= 0  
    BoundaryPrior_output [np.where( BoundaryPrior >0) ]=255
    BoundaryPrior_output = BoundaryPrior_output.astype(np.uint8) 
    return BoundaryPrior_output,Objects_first_few  

def ensure_trailing_slash(path):
    return path.rstrip("/") + "/"


def process_domain(domain_root, mask_generator):
    images_dir = os.path.join(domain_root, "images_png")
    boundary_dir = os.path.join(domain_root, "boundary_pngs")
    object_dir = os.path.join(domain_root, "object_pngs")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_png not found: {images_dir}")

    os.makedirs(boundary_dir, exist_ok=True)
    os.makedirs(object_dir, exist_ok=True)

    img_list = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    print(f"Domain root: {domain_root}")
    print(f"Found images: {len(img_list)}")

    start_time = time.time()
    for i, img_input in enumerate(img_list, 1):
        img_name = img_input.split(".")[0]
        image = np.array(Image.open(os.path.join(images_dir, img_input)))

        result = SAMAug(image, mask_generator)
        if result is None:
            # Fallback to all-zero maps when SAM returns no masks.
            h, w = image.shape[:2]
            boundary = np.zeros((h, w), dtype=np.uint8)
            objects = np.zeros((h, w), dtype=np.uint8)
        else:
            boundary, objects = result
            objects = objects.astype(np.uint8)

        Image.fromarray(boundary).save(os.path.join(boundary_dir, f"{img_name}_Boundary.png"))
        Image.fromarray(objects).save(os.path.join(object_dir, f"{img_name}_objects.png"))

        if i % 50 == 0 or i == len(img_list):
            print(f"[{i}/{len(img_list)}] processed")

    run_time = time.time() - start_time
    print(f"Finished {domain_root} in {run_time:.2f}s")


if __name__ == "__main__":
    # Example:
    # LOVEDA_ROOT=/root/FPAA/dataset/LoveDA/Train python SAM_utils.py
    loveDA_root = os.environ.get("LOVEDA_ROOT", "./loveDA/Train")
    loveDA_root = ensure_trailing_slash(loveDA_root)
    domains = os.environ.get("LOVEDA_DOMAINS", "Urban").split(",")
    domains = [d.strip() for d in domains if d.strip()]

    print(f"LOVEDA_ROOT={loveDA_root}")
    print(f"LOVEDA_DOMAINS={domains}")

    for domain in domains:
        process_domain(os.path.join(loveDA_root, domain), mask_generator)