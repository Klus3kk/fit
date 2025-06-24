"""
Test Kaggle-specific features: preprocessing, validation, and high-level APIs.
"""

import numpy as np
from fit.core.tensor import Tensor

def test_preprocessing():
    """Test data preprocessing utilities."""
    print("=== Testing Data Preprocessing ===")
    
    try:
        from fit.data.preprocessing import StandardScaler, MinMaxScaler
        
        # Create sample data
        X = np.random.randn(100, 4) * 10 + 5
        print(f"Original data shape: {X.shape}")
        print(f"Original data mean: {X.mean(axis=0)}")
        print(f"Original data std: {X.std(axis=0)}")
        
        # Test StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Scaled data mean: {X_scaled.mean(axis=0)}")
        print(f"Scaled data std: {X_scaled.std(axis=0)}")
        
        # Test inverse transform
        X_inverse = scaler.inverse_transform(X_scaled)
        print(f"Inverse transform error: {np.mean(np.abs(X - X_inverse))}")
        
        # Test MinMaxScaler
        minmax = MinMaxScaler()
        X_minmax = minmax.fit_transform(X)
        print(f"MinMax scaled range: [{X_minmax.min():.3f}, {X_minmax.max():.3f}]")
        
        return True
    except ImportError as e:
        print(f"Preprocessing utilities not available: {e}")
        return False

def test_cross_validation():
    """Test cross-validation utilities."""
    print("\n=== Testing Cross-Validation ===")
    
    try:
        from fit.data.validation import KFoldCV, StratifiedKFoldCV
        
        # Create sample data
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 3, 50)
        
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test KFold
        kfold = KFoldCV(n_splits=5, shuffle=True, random_state=42)
        
        print("K-Fold Cross-Validation:")
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"Fold {fold + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        # Test Stratified KFold
        stratified = StratifiedKFoldCV(n_splits=3, shuffle=True, random_state=42)
        
        print("\nStratified K-Fold Cross-Validation:")
        for fold, (train_idx, val_idx) in enumerate(stratified.split(X, y)):
            train_y = y[train_idx]
            val_y = y[val_idx]
            print(f"Fold {fold + 1}: Train classes {np.bincount(train_y)}, Val classes {np.bincount(val_y)}")
        
        return True
    except ImportError as e:
        print(f"Cross-validation utilities not available: {e}")
        return False

def test_kaggle_specific():
    """Test Kaggle-specific utilities."""
    print("\n=== Testing Kaggle-Specific Features ===")
    
    try:
        from fit.data.validation import kaggle_time_series_split, kaggle_group_split
        
        # Test time series split
        print("Time Series Split:")
        time_data = np.random.randn(100, 3)
        time_column = np.arange(100)  # Time column
        
        # Add time column to data
        data_with_time = np.column_stack([time_data, time_column])
        
        train_idx, test_idx = kaggle_time_series_split(
            data_with_time, 
            time_column_idx=3,  # Time is last column
            test_size=0.2,
            gap_size=5
        )
        
        print(f"Train samples: {len(train_idx)}")
        print(f"Test samples: {len(test_idx)}")
        print(f"Gap maintained: {min(test_idx) - max(train_idx) >= 5}")
        
        # Test group split
        print("\nGroup Split:")
        X = np.random.randn(100, 4)
        groups = np.random.randint(0, 10, 100)  # 10 different groups
        
        train_idx, test_idx = kaggle_group_split(X, groups, test_size=0.3)
        
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        
        print(f"Train groups: {len(train_groups)}")
        print(f"Test groups: {len(test_groups)}")
        print(f"No group overlap: {len(train_groups & test_groups) == 0}")
        
        return True
    except ImportError as e:
        print(f"Kaggle-specific utilities not available: {e}")
        return False

def test_simple_apis():
    """Test high-level simple APIs."""
    print("\n=== Testing Simple APIs ===")
    
    try:
        from fit.simple.trainer import SimpleTrainer
        from fit.simple.models import MLP
        
        print("Testing simple model creation...")
        
        # Create a simple MLP
        model = MLP([4, 8, 3], activation='relu')
        print("MLP created successfully!")
        
        # Create simple data
        X = np.random.randn(20, 4)
        y = np.random.randint(0, 3, 20)
        
        print("Simple APIs available!")
        return True
        
    except ImportError as e:
        print(f"Simple APIs not available: {e}")
        return False

def test_advanced_features():
    """Test advanced features like SAM optimizer."""
    print("\n=== Testing Advanced Features ===")
    
    try:
        from fit.optim.experimental.sam import SAM
        from fit.optim.adam import Adam
        from fit.nn.modules.linear import Linear
        
        # Create a simple model
        model_params = [Linear(2, 1).weight, Linear(2, 1).bias]
        
        # Create base optimizer
        base_optimizer = Adam(model_params, lr=0.01)
        
        # Create SAM optimizer
        sam = SAM(model_params, base_optimizer, rho=0.05)
        
        print("SAM optimizer created successfully!")
        print(f"SAM rho parameter: {sam.rho}")
        
        return True
    except ImportError as e:
        print(f"Advanced features not available: {e}")
        return False

if __name__ == "__main__":
    try:
        test_preprocessing()
        test_cross_validation()
        test_kaggle_specific()
        test_simple_apis()
        test_advanced_features()
        print("\n✅ Kaggle feature tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()