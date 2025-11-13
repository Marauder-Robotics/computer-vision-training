#!/usr/bin/env python3
"""
Marauder CV Project - Species Mapping Validation Script
Validates species classifications, ID assignments, and config structure

Usage:
    python scripts/validate_species_mapping.py
    python scripts/validate_species_mapping.py --config config/species_mapping.yaml
    python scripts/validate_species_mapping.py --strict  # Fail on warnings
"""

import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class SpeciesMappingValidator:
    """Validates species mapping configuration"""
    
    def __init__(self, config_path: Path, strict: bool = False):
        self.config_path = config_path
        self.strict = strict
        self.errors = []
        self.warnings = []
        self.species_data = None
        
    def log(self, message: str, level: str = "info"):
        """Log a message with color"""
        colors = {
            "info": "\033[0m",
            "success": "\033[0;32m",
            "error": "\033[0;31m",
            "warning": "\033[1;33m",
        }
        reset = "\033[0m"
        print(f"{colors.get(level, '')}{message}{reset}")
    
    def add_error(self, message: str):
        """Add an error"""
        self.errors.append(message)
        self.log(f"✗ ERROR: {message}", "error")
    
    def add_warning(self, message: str):
        """Add a warning"""
        self.warnings.append(message)
        self.log(f"⚠ WARNING: {message}", "warning")
    
    def load_config(self) -> bool:
        """Load and parse species mapping config"""
        self.log(f"\n=== Loading config from {self.config_path} ===\n")
        
        try:
            if not self.config_path.exists():
                self.add_error(f"Config file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.species_data = yaml.safe_load(f)
            
            if not self.species_data:
                self.add_error("Config file is empty")
                return False
            
            if 'species' not in self.species_data:
                self.add_error("Config missing 'species' key")
                return False
            
            self.log("✓ Config loaded successfully", "success")
            return True
            
        except yaml.YAMLError as e:
            self.add_error(f"YAML parsing error: {e}")
            return False
        except Exception as e:
            self.add_error(f"Failed to load config: {e}")
            return False
    
    def validate_structure(self) -> bool:
        """Validate config structure"""
        self.log("\n=== Test 1: Config Structure ===\n")
        
        species = self.species_data.get('species', {})
        
        # Check required categories
        required_categories = ['critical', 'important', 'general']
        for category in required_categories:
            if category not in species:
                self.add_error(f"Missing required category: '{category}'")
            else:
                self.log(f"✓ Found category: {category}", "success")
        
        # Check that categories are lists
        for category in required_categories:
            if category in species:
                if not isinstance(species[category], list):
                    self.add_error(f"Category '{category}' must be a list")
        
        return len(self.errors) == 0
    
    def validate_species_counts(self) -> bool:
        """Validate species counts per category"""
        self.log("\n=== Test 2: Species Counts ===\n")
        
        species = self.species_data['species']
        
        # Expected counts from project spec
        expected_counts = {
            'critical': 20,
            'important': 9,
            'general': 7
        }
        
        all_valid = True
        
        for category, expected_count in expected_counts.items():
            if category not in species:
                continue
                
            actual_count = len(species[category])
            
            if actual_count == expected_count:
                self.log(
                    f"✓ {category.capitalize()}: {actual_count} species (expected {expected_count})",
                    "success"
                )
            else:
                self.add_warning(
                    f"{category.capitalize()}: {actual_count} species (expected {expected_count})"
                )
                all_valid = False
        
        # Total species count
        total = sum(len(species[cat]) for cat in ['critical', 'important', 'general'] if cat in species)
        expected_total = 36
        
        self.log(f"\nTotal species: {total}", "info")
        
        if total != expected_total:
            self.add_warning(f"Total species is {total}, expected {expected_total}")
        
        return all_valid
    
    def validate_id_assignments(self) -> bool:
        """Validate species ID assignments"""
        self.log("\n=== Test 3: ID Assignments ===\n")
        
        species = self.species_data['species']
        
        # Expected ID ranges
        id_ranges = {
            'critical': (0, 19),   # IDs 0-19
            'important': (20, 28),  # IDs 20-28
            'general': (29, 35)     # IDs 29-35
        }
        
        all_ids = set()
        all_valid = True
        
        for category, (min_id, max_id) in id_ranges.items():
            if category not in species:
                continue
            
            category_species = species[category]
            category_ids = []
            
            for sp in category_species:
                if 'id' not in sp:
                    self.add_error(f"Species missing ID in {category}: {sp.get('common_name', 'unknown')}")
                    all_valid = False
                    continue
                
                sp_id = sp['id']
                category_ids.append(sp_id)
                
                # Check ID in valid range
                if not (min_id <= sp_id <= max_id):
                    self.add_error(
                        f"ID {sp_id} out of range for {category} "
                        f"(expected {min_id}-{max_id}): {sp.get('common_name', 'unknown')}"
                    )
                    all_valid = False
                
                # Check for duplicate IDs
                if sp_id in all_ids:
                    self.add_error(f"Duplicate ID {sp_id}: {sp.get('common_name', 'unknown')}")
                    all_valid = False
                else:
                    all_ids.add(sp_id)
            
            # Check for sequential IDs (warning only)
            if category_ids:
                sorted_ids = sorted(category_ids)
                expected_ids = list(range(min_id, min_id + len(category_ids)))
                if sorted_ids != expected_ids:
                    self.add_warning(
                        f"{category.capitalize()} IDs not sequential: {sorted_ids}"
                    )
        
        if all_valid:
            self.log(f"✓ All IDs valid and unique", "success")
            self.log(f"  Total unique IDs: {len(all_ids)}", "info")
        
        return all_valid
    
    def validate_required_fields(self) -> bool:
        """Validate required fields for each species"""
        self.log("\n=== Test 4: Required Fields ===\n")
        
        species = self.species_data['species']
        
        required_fields = ['id', 'common_name', 'scientific_name', 'category']
        optional_fields = ['alert_priority', 'fathomnet_concepts']
        
        all_valid = True
        field_counts = defaultdict(int)
        
        for category in ['critical', 'important', 'general']:
            if category not in species:
                continue
            
            for sp in species[category]:
                # Check required fields
                for field in required_fields:
                    if field not in sp:
                        self.add_error(
                            f"Missing required field '{field}' for {sp.get('common_name', 'unknown')}"
                        )
                        all_valid = False
                    elif sp[field] is None or (isinstance(sp[field], str) and not sp[field]):
                        # Allow id=0 but catch None or empty strings
                        self.add_error(
                            f"Empty value for required field '{field}' for {sp.get('common_name', 'unknown')}"
                        )
                        all_valid = False
                    else:
                        field_counts[field] += 1
                
                # Check category matches
                if sp.get('category') != category:
                    self.add_error(
                        f"Category mismatch for {sp.get('common_name', 'unknown')}: "
                        f"in '{category}' list but category='{sp.get('category')}'"
                    )
                    all_valid = False
        
        if all_valid:
            self.log(f"✓ All species have required fields", "success")
            for field, count in sorted(field_counts.items()):
                self.log(f"  {field}: {count} species", "info")
        
        return all_valid
    
    def validate_critical_species_alerts(self) -> bool:
        """Validate critical species have alert_priority"""
        self.log("\n=== Test 5: Critical Species Alerts ===\n")
        
        species = self.species_data['species']
        
        if 'critical' not in species:
            return True
        
        all_valid = True
        
        for sp in species['critical']:
            if 'alert_priority' not in sp:
                self.add_warning(
                    f"Critical species missing alert_priority: {sp.get('common_name', 'unknown')}"
                )
                all_valid = False
            elif sp['alert_priority'] != 'immediate':
                self.add_warning(
                    f"Critical species should have alert_priority='immediate': "
                    f"{sp.get('common_name', 'unknown')} has '{sp['alert_priority']}'"
                )
        
        if all_valid:
            self.log(f"✓ All critical species have immediate alerts", "success")
        
        return all_valid
    
    def validate_fathomnet_concepts(self) -> bool:
        """Validate FathomNet concept mappings"""
        self.log("\n=== Test 6: FathomNet Concepts ===\n")
        
        species = self.species_data['species']
        
        species_with_concepts = 0
        species_without_concepts = []
        
        for category in ['critical', 'important', 'general']:
            if category not in species:
                continue
            
            for sp in species[category]:
                if 'fathomnet_concepts' in sp and sp['fathomnet_concepts']:
                    species_with_concepts += 1
                    # Validate it's a list
                    if not isinstance(sp['fathomnet_concepts'], list):
                        self.add_error(
                            f"fathomnet_concepts must be a list for {sp.get('common_name', 'unknown')}"
                        )
                else:
                    species_without_concepts.append(
                        f"{sp.get('common_name', 'unknown')} ({category})"
                    )
        
        self.log(f"✓ {species_with_concepts} species have FathomNet concepts", "success")
        
        if species_without_concepts:
            self.add_warning(
                f"{len(species_without_concepts)} species without FathomNet concepts"
            )
            if len(species_without_concepts) <= 5:
                for sp_name in species_without_concepts:
                    self.log(f"  - {sp_name}", "warning")
        
        return True  # Just a warning, not an error
    
    def validate_name_uniqueness(self) -> bool:
        """Validate species names are unique"""
        self.log("\n=== Test 7: Name Uniqueness ===\n")
        
        species = self.species_data['species']
        
        common_names = set()
        scientific_names = set()
        duplicates = []
        
        for category in ['critical', 'important', 'general']:
            if category not in species:
                continue
            
            for sp in species[category]:
                common = sp.get('common_name', '').lower()
                scientific = sp.get('scientific_name', '').lower()
                
                if common in common_names:
                    duplicates.append(f"Duplicate common name: {sp.get('common_name')}")
                else:
                    common_names.add(common)
                
                if scientific in scientific_names:
                    duplicates.append(f"Duplicate scientific name: {sp.get('scientific_name')}")
                else:
                    scientific_names.add(scientific)
        
        if duplicates:
            for dup in duplicates:
                self.add_error(dup)
            return False
        else:
            self.log(f"✓ All species names are unique", "success")
            self.log(f"  {len(common_names)} unique common names", "info")
            self.log(f"  {len(scientific_names)} unique scientific names", "info")
            return True
    
    def run_all_validations(self) -> bool:
        """Run all validation tests"""
        self.log("=" * 60)
        self.log("Marauder CV - Species Mapping Validation")
        self.log("=" * 60)
        
        # Load config
        if not self.load_config():
            return False
        
        # Run validation tests
        tests = [
            self.validate_structure,
            self.validate_species_counts,
            self.validate_id_assignments,
            self.validate_required_fields,
            self.validate_critical_species_alerts,
            self.validate_fathomnet_concepts,
            self.validate_name_uniqueness,
        ]
        
        for test in tests:
            test()
        
        # Print summary
        self.log("\n" + "=" * 60)
        self.log("Validation Summary")
        self.log("=" * 60)
        
        if not self.errors:
            self.log("✓ No errors found", "success")
        else:
            self.log(f"✗ {len(self.errors)} error(s) found:", "error")
            for error in self.errors:
                self.log(f"  - {error}", "error")
        
        if self.warnings:
            self.log(f"\n⚠ {len(self.warnings)} warning(s):", "warning")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                self.log(f"  - {warning}", "warning")
            if len(self.warnings) > 5:
                self.log(f"  ... and {len(self.warnings) - 5} more", "warning")
        
        self.log("")
        
        # Determine success
        has_errors = len(self.errors) > 0
        has_warnings = len(self.warnings) > 0
        
        if not has_errors and not has_warnings:
            self.log("✓ All validations passed!", "success")
            return True
        elif not has_errors and has_warnings:
            if self.strict:
                self.log("✗ Warnings treated as errors in strict mode", "error")
                return False
            else:
                self.log("✓ Validation passed with warnings", "success")
                return True
        else:
            self.log("✗ Validation failed", "error")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate species mapping configuration for Marauder CV project"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/species_mapping.yaml'),
        help='Path to species mapping config file (default: config/species_mapping.yaml)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = SpeciesMappingValidator(args.config, strict=args.strict)
    success = validator.run_all_validations()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
