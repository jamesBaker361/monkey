import gdown
import os

file_id = "1RYKdbjEfeRovyfH4iiautbfl6yh2-GEs"
output = "cuhk_sysu.zip"  # change the name as needed

if not os.path.exists(output):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Source - https://stackoverflow.com/a/3451150
# Posted by Rahul, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-30, License - CC BY-SA 4.0

import zipfile
directory="cuhk_sysu"
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(directory)
