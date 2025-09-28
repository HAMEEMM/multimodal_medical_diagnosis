
"""
Phase 5 Model Training - Variable Loading Script (CORRECTED)
Generated automatically from Phase 4 Step 15 (Corrected Version)

This script loads all available variables for Phase 5 model training.
Variables have been verified for availability and consistency.
"""

import joblib
import os

# Project directory
project_dir = r"G:\Msc\NCU\Doctoral Record\multimodal_medical_diagnosis"
phase4_step15_dir = os.path.join(project_dir, 'variables', 'phase4_step15_audio_text')

print("🚀 LOADING AVAILABLE PHASE 5 VARIABLES...")

# Check and load available variables
try:
    # Load training data
    X_train_path = os.path.join(phase4_step15_dir, 'X_train_phase5.joblib')
    if os.path.exists(X_train_path):
        X_train = joblib.load(X_train_path)
        print(f"   📂 Loaded X_train: {X_train.shape}")
    else:
        print("   ❌ X_train not available")
        X_train = None

    y_train_path = os.path.join(phase4_step15_dir, 'y_train_phase5.joblib')
    if os.path.exists(y_train_path):
        y_train = joblib.load(y_train_path)
        print(f"   📂 Loaded y_train: {y_train.shape}")
    else:
        print("   ❌ y_train not available")
        y_train = None

    # Load validation data
    X_val_path = os.path.join(phase4_step15_dir, 'X_val_phase5.joblib')
    if os.path.exists(X_val_path):
        X_val = joblib.load(X_val_path)
        print(f"   📂 Loaded X_val: {X_val.shape}")
    else:
        print("   ❌ X_val not available")
        X_val = None

    y_val_path = os.path.join(phase4_step15_dir, 'y_val_phase5.joblib')
    if os.path.exists(y_val_path):
        y_val = joblib.load(y_val_path)
        print(f"   📂 Loaded y_val: {y_val.shape}")
    else:
        print("   ❌ y_val not available")
        y_val = None

    # Load testing data
    X_test_path = os.path.join(phase4_step15_dir, 'X_test_phase5.joblib')
    if os.path.exists(X_test_path):
        X_test = joblib.load(X_test_path)
        print(f"   📂 Loaded X_test: {X_test.shape}")
    else:
        print("   ❌ X_test not available")
        X_test = None

    y_test_path = os.path.join(phase4_step15_dir, 'y_test_phase5.joblib')
    if os.path.exists(y_test_path):
        y_test = joblib.load(y_test_path)
        print(f"   📂 Loaded y_test: {y_test.shape}")
    else:
        print("   ❌ y_test not available")
        y_test = None

    # Load label encoder
    label_encoder_path = os.path.join(phase4_step15_dir, 'label_encoder_phase5.joblib')
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
        print(f"   📂 Loaded label_encoder: {len(label_encoder.classes_)} classes")
    else:
        print("   ❌ label_encoder not available")
        label_encoder = None

    # Load preprocessing pipeline
    pipeline_path = os.path.join(phase4_step15_dir, 'deployment_pipeline_phase5.joblib')
    if os.path.exists(pipeline_path):
        deployment_pipeline = joblib.load(pipeline_path)
        print(f"   📂 Loaded deployment_pipeline: {len(deployment_pipeline)} components")
    else:
        print("   ❌ deployment_pipeline not available")
        deployment_pipeline = None

    # Verify essential components
    essential_available = all([X_train is not None, y_train is not None, X_val is not None, 
                              y_val is not None, X_test is not None, y_test is not None, 
                              label_encoder is not None])

    print("\n✅ VERIFICATION:")
    if essential_available:
        print("   🎯 All essential variables loaded successfully!")
        print(f"   Training set: X_train {{X_train.shape}}, y_train {{y_train.shape}}")
        print(f"   Validation set: X_val {{X_val.shape}}, y_val {{y_val.shape}}")
        print(f"   Testing set: X_test {{X_test.shape}}, y_test {{y_test.shape}}")
        print(f"   Classes: {{len(label_encoder.classes_)}} medical conditions")
        print(f"   Features: {{X_train.shape[1]}} dimensions")
        print(f"   Total samples: {{X_train.shape[0] + X_val.shape[0] + X_test.shape[0]:,}}")
        print("\n🎯 READY FOR PHASE 5 MODEL TRAINING!")
    else:
        print("   ⚠️ Some essential variables are missing - check availability before training")
        print("\n⚠️ PARTIAL READINESS FOR PHASE 5")

except Exception as e:
    print(f"❌ Error loading variables: {str(e)}")
