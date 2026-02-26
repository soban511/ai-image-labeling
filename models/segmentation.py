# import os
# import sys
# import torch
# import numpy as np
# import cv2
# from PIL import Image
# from torchvision import transforms

# # Path setup
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# repo_path = os.path.join(BASE_DIR, "U-2-Net")
# sys.path.append(repo_path)

# from model.u2net import U2NETP

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load model ONCE
# model_path = os.path.join(repo_path, "u2net.pth")
# u2net_model = U2NETP(3, 1)
# u2net_model.load_state_dict(torch.load(model_path, map_location=device))
# u2net_model.to(device).eval()


# def segment_and_crop(image_path, padding=0):
#     image = Image.open(image_path).convert("RGB")
#     orig = np.array(image)

#     transform = transforms.Compose([
#         transforms.Resize((320, 320)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])

#     inp = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         pred = u2net_model(inp)[0]
#         mask = torch.sigmoid(pred).cpu().numpy()[0, 0]

#     mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
#     mask = (mask * 255).astype(np.uint8)
#     _, bin_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

#     coords = cv2.findNonZero(bin_mask)
#     if coords is None:
#         raise ValueError("No foreground object detected.")

#     x, y, w, h = cv2.boundingRect(coords)

#     x1 = max(0, x - padding)
#     y1 = max(0, y - padding)
#     x2 = min(orig.shape[1], x + w + padding)
#     y2 = min(orig.shape[0], y + h + padding)

#     cropped = orig[y1:y2, x1:x2]
#     return cropped



from rembg import remove
from PIL import Image
import numpy as np
import io


def segment_and_crop(image_path, debug=False):
    """
    Removes background using rembg,
    finds bounding box from mask,
    crops region from ORIGINAL image.

    Returns:
        cropped_original (numpy array)
    """

    # Load original image
    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original)

    # Remove background
    with open(image_path, "rb") as f:
        input_bytes = f.read()

    output_bytes = remove(input_bytes)

    # Get RGBA output to extract mask
    rgba_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    rgba_np = np.array(rgba_image)

    # Extract alpha channel as mask
    alpha = rgba_np[:, :, 3]

    coords = np.column_stack(np.where(alpha > 0))
    if coords.size == 0:
        raise ValueError("No object detected.")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # Crop from ORIGINAL image
    cropped_original = original_np[y0:y1, x0:x1]

    if debug:
        print("Original shape:", original_np.shape)
        print("Mask shape:", alpha.shape)
        print("Bounding box:", (y0, x0, y1, x1))
        print("Cropped shape:", cropped_original.shape)

    return cropped_original