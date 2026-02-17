#!/usr/bin/env python
"""Test script to verify minimal dependencies work for GRADIEND."""

import sys
sys.path.insert(0, '.')

print("Testing GRADIEND imports with minimal dependencies...")
print("=" * 60)

try:
    print("\n1. Testing core model imports...")
    from gradiend.model import GradiendModel, ModelWithGradiend
    print("   ✓ Core model classes imported")
    
    print("\n2. Testing training imports...")
    from gradiend.trainer import (
        load_training_stats,
        TrainingArguments,
        TextGradientTrainingDataset,
        create_model_with_gradiend,
    )
    print("   ✓ Training imports (load_training_stats, TrainingArguments, ...)")
    
    print("\n3. Testing definition imports...")
    from gradiend.trainer.core.feature_definition import FeatureLearningDefinition
    from gradiend.trainer.core.protocols import DataProvider, Evaluator, FeatureAnalyzer
    print("   ✓ FeatureLearningDefinition and protocols imported")
    
    print("\n4. Testing utility imports...")
    from gradiend.util import hash_it, convert_tuple_keys_to_strings
    print("   ✓ Utility functions imported")
    
    print("\n" + "=" * 60)
    print("✅ All imports successful!")
    print("✅ Minimal dependencies are sufficient for GRADIEND core functionality")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nThis might indicate a missing dependency.")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"\n⚠️  Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
