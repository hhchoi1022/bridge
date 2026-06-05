#%%
import json
import yaml
from pathlib import Path
#%%
config_dir = Path("./")  # change if needed

for json_path in config_dir.glob("*.config"):
    yaml_path = json_path.with_suffix(".config")

    with open(json_path, "r") as f:
        data = json.load(f)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"✅ Converted: {json_path.name} → {yaml_path.name}")

# %%
