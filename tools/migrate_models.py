"""Migrate existing models to versioned naming convention.

This script:
1. Creates weights/ subdirectories
2. Renames models to versioned format: {name}_v1.0.0_{date}.pt
3. Updates config.yaml files with model_version field
4. Creates backup before migration
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

# Migration date
MIGRATION_DATE = datetime.now().strftime("%Y%m%d")
INITIAL_VERSION = "1.0.0"

# Model migration map: (product, area, old_filename) -> new_base_name
MODEL_MIGRATIONS = [
    ("Cable1", "A", "best.pt", "Cable1_best"),
    ("LED", "A", "LED_best.pt", "LED_best"),
    ("LED", "A", "LED_color_mixing_best.pt", "LED_color_mixing"),
    ("LED", "B", "LED_best.pt", "LED_best"),
    ("LED", "B", "LED_color_mixing_best.pt", "LED_color_mixing"),
    ("LED", "C", "LED_best.pt", "LED_best"),
    ("LED", "C", "LED_color_mixing_best.pt", "LED_color_mixing"),
    ("LED", "D", "LED_best.pt", "LED_best"),
    ("LED", "D", "LED_color_mixing_best.pt", "LED_color_mixing"),
    ("PCBA1", "A", "PCBA_A.pt", "PCBA1_A"),
    ("PCBA1", "B", "PCBA_B.pt", "PCBA1_B"),
    ("PCBA1", "C", "PCBA_C.pt", "PCBA1_C"),
    ("PCBA2", "B", "PCBA_B.pt", "PCBA2_B"),
    ("PCBA3", "A", "best.pt", "PCBA3_best"),
    ("PCBA3", "A", "PCBA_B.pt", "PCBA3_B"),
]


def migrate_model(
    product: str, area: str, old_filename: str, new_base_name: str, dry_run: bool = False
) -> bool:
    """Migrate a single model file.
    
    Args:
        product: Product name
        area: Area name
        old_filename: Current filename
        new_base_name: New base name (without version/extension)
        dry_run: If True, only print actions without executing
    
    Returns:
        True if migration successful
    """
    models_dir = Path("models")
    yolo_dir = models_dir / product / area / "yolo"
    
    old_path = yolo_dir / old_filename
    if not old_path.exists():
        print(f"SKIP: {old_path} not found")
        return False
    
    # Create weights/ subdirectory
    weights_dir = yolo_dir / "weights"
    if not dry_run:
        weights_dir.mkdir(exist_ok=True)
    
    # Generate new filename
    new_filename = f"{new_base_name}_v{INITIAL_VERSION}_{MIGRATION_DATE}.pt"
    new_path = weights_dir / new_filename
    
    print(f"[{product}/{area}/yolo]")
    print(f"   {old_filename} -> weights/{new_filename}")
    
    if not dry_run:
        shutil.move(str(old_path), str(new_path))
        print(f"   OK: Moved")
    else:
        print(f"   DRY RUN - would move")
    
    return True


def update_config(product: str, area: str, dry_run: bool = False) -> bool:
    """Update config.yaml with model_version field.
    
    Args:
        product: Product name
        area: Area name
        dry_run: If True, only print actions
    
    Returns:
        True if update successful
    """
    config_path = Path("models") / product / area / "yolo" / "config.yaml"
    
    if not config_path.exists():
        print(f"WARNING: Config not found: {config_path}")
        return False
    
    # Read existing config
    with open(config_path, encoding="utf-8") as f:
        content = f.read()
    
    # Check if already has model_version
    if "model_version:" in content:
        print(f"   INFO: {product}/{area} config already has model_version")
        return True
    
    # Add model_version at the end
    new_content = content.rstrip() + f"\n\n# Model version (added by migration)\nmodel_version: \"{INITIAL_VERSION}\"\n"
    
    print(f"   Updating {product}/{area}/yolo/config.yaml")
    
    if not dry_run:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"   OK: Added model_version: {INITIAL_VERSION}")
    else:
        print(f"   DRY RUN - would add model_version")
    
    return True


def create_backup(dry_run: bool = False) -> None:
    """Create backup of models/ directory."""
    backup_dir = Path("models_backup")
    
    if backup_dir.exists():
        print(f"WARNING: Backup already exists: {backup_dir}")
        return
    
    print(f"Creating backup: models/ -> models_backup/")
    
    if not dry_run:
        shutil.copytree("models", "models_backup")
        print(f"OK: Backup created")
    else:
        print(f"DRY RUN - would create backup")


def main(dry_run: bool = False) -> None:
    """Main migration function.
    
    Args:
        dry_run: If True, simulate migration without making changes
    """
    print("=" * 70)
    print("Model Version Migration Script")
    print("=" * 70)
    print(f"Migration Date: {MIGRATION_DATE}")
    print(f"Initial Version: {INITIAL_VERSION}")
    print(f"Models to migrate: {len(MODEL_MIGRATIONS)}")
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
    print("=" * 70)
    print()
    
    # Step 1: Create backup
    print("Step 1: Creating backup...")
    create_backup(dry_run)
    print()
    
    # Step 2: Migrate models
    print("Step 2: Migrating models...")
    success_count = 0
    for product, area, old_name, new_base in MODEL_MIGRATIONS:
        if migrate_model(product, area, old_name, new_base, dry_run):
            success_count += 1
    print(f"\nMigrated {success_count}/{len(MODEL_MIGRATIONS)} models")
    print()
    
    # Step 3: Update configs
    print("Step 3: Updating config files...")
    unique_products = set((p, a) for p, a, _, _ in MODEL_MIGRATIONS)
    config_count = 0
    for product, area in unique_products:
        if update_config(product, area, dry_run):
            config_count += 1
    print(f"\nUpdated {config_count}/{len(unique_products)} configs")
    print()
    
    # Summary
    print("=" * 70)
    if dry_run:
        print("DRY RUN COMPLETE - No changes were made")
        print("Run without --dry-run to apply changes:")
        print("   python tools/migrate_models.py")
    else:
        print("MIGRATION COMPLETE!")
        print()
        print("Next steps:")
        print("   1. Verify models loaded correctly:")
        print("      pytest tests/test_version_utils.py -v")
        print("   2. Test inference:")
        print("      python main.py --product LED --area A --type yolo")
        print("   3. Commit changes:")
        print("      git add models/")
        print("      git commit -m 'refactor: migrate models to versioned naming'")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # Check for dry-run flag
    is_dry_run = "--dry-run" in sys.argv
    
    try:
        main(dry_run=is_dry_run)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
