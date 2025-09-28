
# Phase 5 Loading Script Template
# Generated automatically from Phase 4 variable inventory


# Project directory
project_dir = r"G:\Msc\NCU\Doctoral Record\multimodal_medical_diagnosis"

# Essential variables for Phase 5 (update paths as needed)

# Load final_normalized_test_features
final_normalized_test_features_path = os.path.join(project_dir, r"variables\phase4_step13_audio_text\final_normalized_test_features.joblib")
if os.path.exists(final_normalized_test_features_path):
    final_normalized_test_features = joblib.load(final_normalized_test_features_path)
    print(f"✅ Loaded final_normalized_test_features: {type(final_normalized_test_features).__name__}")
else:
    print(f"❌ File not found: {final_normalized_test_features_path}")


# Verify essential variables are loaded
essential_variables = [
    'final_normalized_test_features',
]

print("\n🔍 VERIFICATION:")
for var_name in essential_variables:
    if var_name in locals():
        var_obj = locals()[var_name]
        if hasattr(var_obj, 'shape'):
            print(f"   ✅ {var_name}: {type(var_obj).__name__} {var_obj.shape}")
        elif hasattr(var_obj, '__len__'):
            print(f"   ✅ {var_name}: {type(var_obj).__name__} length {len(var_obj)}")
        else:
            print(f"   ✅ {var_name}: {type(var_obj).__name__}")
    else:
        print(f"   ❌ {var_name}: NOT LOADED")

print("\n🚀 Ready for Phase 5 model training!")
