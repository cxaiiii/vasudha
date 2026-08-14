import modal

# Connect directly to your persistent Modal Volume
vol = modal.Volume.from_name("vasudha-vol")

# Stream the remote weights file directly into a local file path
with open("vasudha-model/vasudha/model.safetensors-00001-of-00002.safetensors", "wb") as f:
    for chunk in vol.read_file("ckpt/models/vasudha/model.safetensors-00001-of-00002.safetensors"):
        f.write(chunk)
        print(f"Downloaded {f.tell()} bytes...", end="\r")

print("Download 100% complete!")
