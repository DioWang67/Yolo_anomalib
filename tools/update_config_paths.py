"""Update config.yaml weights paths to versioned models.

This script updates all config.yaml files to point to the new versioned
model files in the weights/ subdirectory.
"""

from pathlib import Path
import yaml

# Mapping: (product, area) -> (old_weight_name, new_versioned_name)
CONFIG_UPDATES = {
    ("Cable1", "A"): ("best.pt", "weights/Cable1_best_v1.0.0_20260210.pt"),
    ("LED", "A"): ("LED_color_mixing_best.pt", "weights/LED_color_mixing_v1.0.0_20260210.pt"),
    ("LED", "B"): ("LED_color_mixing_best.pt", "weights/LED_color_mixing_v1.0.0_20260210.pt"),
    ("LED", "C"): ("LED_color_mixing_best.pt", "weights/LED_color_mixing_v1.0.0_20260210.pt"),
    ("LED", "D"): ("LED_color_mixing_best.pt", "weights/LED_color_mixing_v1.0.0_20260210.pt"),
    ("PCBA1", "A"): ("PCBA_A.pt", "weights/PCBA1_A_v1.0.0_20260210.pt"),
    ("PCBA1", "B"): ("PCBA_B.pt", "weights/PCBA1_B_v1.0.0_20260210.pt"),
    ("PCBA1", "C"): ("PCBA_C.pt", "weights/PCBA1_C_v1.0.0_20260210.pt"),
    ("PCBA2", "B"): ("PCBA_B.pt", "weights/PCBA2_B_v1.0.0_20260210.pt"),
    ("PCBA3", "A"): ("PCBA_B.pt", "weights/PCBA3_B_v1.0.0_20260210.pt"),
}


def update_config_weights(product: str, area: str, old_name: str, new_path: str) -> bool:
    """Update weights path in config.yaml.
    
    Args:
        product: Product name
        area: Area name
        old_name: Old weight filename (for verification)
        new_path: New relative path (e.g., "weights/LED_best_v1.0.0_20260210.pt")
    
    Returns:
        True if updated successfully
    """
    config_path = Path("models") / product / area / "yolo" / "config.yaml"
    
    if not config_path.exists():
        print(f"WARNING: Config not found: {config_path}")
        return False
    
    # Read config
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if not config:
        print(f"WARNING: Empty config: {config_path}")
        return False
    
    # Get current weights path
    current_weights = config.get("weights", "")
    
    # Build new absolute path
    base_dir = config_path.parent
    new_absolute = (base_dir / new_path).resolve()
    new_absolute_str = str(new_absolute).replace("\\", "/")
    
    # Update weights path
    config["weights"] = new_absolute_str
    
    # Write back
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"[{product}/{area}]")
    print(f"   OLD: {Path(current_weights).name}")
    print(f"   NEW: {new_path}")
    print(f"   OK: Updated")
    
    return True


def main():
    """Update all config files."""
    print("=" * 70)
    print("Updating Config Weights Paths")
    print("=" * 70)
    print()
    
    success_count = 0
    for (product, area), (old_name, new_path) in CONFIG_UPDATES.items():
        if update_config_weights(product, area, old_name, new_path):
            success_count += 1
        print()
    
    print("=" * 70)
    print(f"Updated {success_count}/{len(CONFIG_UPDATES)} configs")
    print("=" * 70)


if __name__ == "__main__":
    main()
