"""
Test script to verify the IDS system setup and test individual components.
Run this to ensure all modules are properly installed and working.

Usage:
    python test_setup.py
"""

import sys
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("=" * 60)
    print("Testing Module Imports...")
    print("=" * 60)
    
    modules_to_test = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-Learn'),
        ('tensorflow', 'TensorFlow'),
        ('PyQt5', 'PyQt5'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    all_passed = True
    for module, name in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED: {e}")
            all_passed = False
    
    print()
    return all_passed

def test_local_modules():
    """Test if local IDS modules can be imported."""
    print("=" * 60)
    print("Testing Local Modules...")
    print("=" * 60)
    
    local_modules = [
        'Feature_selection',
        'Preprocessing',
        'Augmentation',
        'models',
        'dimensionality_reduction',
        'hyperparameter_optimization',
        'thresholding',
        'evaluation',
        'training',
        'ui',
    ]
    
    all_passed = True
    for module in local_modules:
        try:
            __import__(module)
            print(f"✓ {module:30s} - OK")
        except ImportError as e:
            print(f"✗ {module:30s} - FAILED: {e}")
            all_passed = False
    
    print()
    return all_passed

def test_feature_selection():
    """Test feature selection methods."""
    print("=" * 60)
    print("Testing Feature Selection Methods...")
    print("=" * 60)
    
    try:
        from Feature_selection import (
            FilterSelector, WrapperSelector, EmbeddedSelector,
            PSOSelector, ACOSelector
        )
        
        # Create dummy data
        X = np.random.randn(100, 30)
        y = np.random.randint(0, 2, 100)
        
        selectors = [
            ('Filter', FilterSelector(k=10)),
            ('Wrapper', WrapperSelector(k=10)),
            ('Embedded', EmbeddedSelector(k=10)),
            ('PSO', PSOSelector(k=10, n_particles=10, n_iter=2)),
            ('ACO', ACOSelector(k=10, n_ants=10, n_iter=2)),
        ]
        
        for name, selector in selectors:
            try:
                selector.fit(X, y)
                X_selected = selector.transform(X)
                print(f"✓ {name:15s} - OK (Selected {X_selected.shape[1]} features)")
            except Exception as e:
                print(f"✗ {name:15s} - FAILED: {e}")
                return False
        
        print()
        return True
    except Exception as e:
        print(f"✗ Feature selection test FAILED: {e}")
        return False

def test_models():
    """Test model architectures."""
    print("=" * 60)
    print("Testing Model Architectures...")
    print("=" * 60)
    
    try:
        from models import ModelFactory
        
        input_dim = 20
        seq_length = 30
        
        models = [
            ('Attention Autoencoder', 'attention_autoencoder'),
            ('CNN+BiLSTM', 'cnn_bilstm'),
            ('TCN', 'tcn'),
        ]
        
        for name, model_type in models:
            try:
                model = ModelFactory.create_model(model_type, input_dim, seq_length)
                ModelFactory.compile_model(model)
                size_mb = ModelFactory.get_model_size(model)
                print(f"✓ {name:25s} - OK (Size: {size_mb:.2f} MB)")
            except Exception as e:
                print(f"✗ {name:25s} - FAILED: {e}")
                return False
        
        print()
        return True
    except Exception as e:
        print(f"✗ Model test FAILED: {e}")
        return False

def test_dimensionality_reduction():
    """Test dimensionality reduction methods."""
    print("=" * 60)
    print("Testing Dimensionality Reduction Methods...")
    print("=" * 60)
    
    try:
        from dimensionality_reduction_pytorch import DimensionalityReducer
        
        X = np.random.randn(100, 30)
        
        methods = [
            ('PCA', 'pca'),
            ('Autoencoder', 'autoencoder'),
        ]
        
        for name, method in methods:
            try:
                reducer = DimensionalityReducer.create_reducer(method, 30, 15)
                if reducer:
                    reducer.fit(X)
                    X_reduced = reducer.transform(X)
                    print(f"✓ {name:20s} - OK (Reduced to {X_reduced.shape[1]} dims)")
                else:
                    print(f"✓ {name:20s} - OK (None method)")
            except Exception as e:
                print(f"✗ {name:20s} - FAILED: {e}")
                return False
        
        print()
        return True
    except Exception as e:
        print(f"✗ Dimensionality reduction test FAILED: {e}")
        return False

def test_evaluation():
    """Test evaluation metrics."""
    print("=" * 60)
    print("Testing Evaluation Metrics...")
    print("=" * 60)
    
    try:
        from evaluation import IDSEvaluator, PerformanceAnalyzer
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        y_proba = np.random.rand(8, 2)
        
        # Test evaluator
        evaluator = IDSEvaluator()
        results = evaluator.evaluate(y_true, y_pred, y_proba)
        print(f"✓ IDSEvaluator        - OK")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - F1-Score: {results['f1_score']:.4f}")
        
        # Test performance analyzer
        analyzer = PerformanceAnalyzer()
        perf = analyzer.analyze(5.0, 50.0, 15, 30)
        print(f"✓ PerformanceAnalyzer - OK")
        print(f"  - Model Size: {perf['model_size_mb']:.2f} MB")
        print(f"  - Inference Time: {perf['inference_time_ms']:.2f} ms")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Evaluation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing():
    """Test preprocessing module (without actual data)."""
    print("=" * 60)
    print("Testing Preprocessing Module...")
    print("=" * 60)
    
    try:
        from Preprocessing import DataPreprocessor, DATASET_CONFIGS
        
        print(f"✓ DataPreprocessor imported - OK")
        print(f"✓ Supported datasets: {len(DATASET_CONFIGS)}")
        for dataset in DATASET_CONFIGS.keys():
            print(f"  - {dataset}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Preprocessing test FAILED: {e}")
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "IDS SYSTEM SETUP VERIFICATION" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Run all tests
    results.append(("Module Imports", test_imports()))
    results.append(("Local Modules", test_local_modules()))
    results.append(("Feature Selection", test_feature_selection()))
    results.append(("Models", test_models()))
    results.append(("Dimensionality Reduction", test_dimensionality_reduction()))
    results.append(("Evaluation", test_evaluation()))
    results.append(("Preprocessing", test_preprocessing()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print()
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 18 + "✓ ALL TESTS PASSED ✓" + " " * 18 + "║")
        print("║" + " " * 12 + "System is ready to use! Run: python main.py" + " " * 5 + "║")
        print("╚" + "=" * 58 + "╝")
        return 0
    else:
        print()
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 10 + "✗ SOME TESTS FAILED - Check errors above ✗" + " " * 4 + "║")
        print("╚" + "=" * 58 + "╝")
        return 1

if __name__ == '__main__':
    sys.exit(main())
