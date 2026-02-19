import os
import gdown

os.chdir("U-2-Net")

if not os.path.exists("u2net.pth"):
    print("Downloading U²-Net weights...")
    gdown.download(
        "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
        "u2net.pth",
        quiet=False
    )
else:
    print("Weights already exist.")