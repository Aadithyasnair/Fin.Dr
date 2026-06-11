"""
FIN.DR — Model Training Master Controller
Runs the preprocessing and model training scripts to generate model checkpoints.
"""
import sys, os

# Ensure the root folder is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_tensorflow import train as train_tf
from training.train_transformer import train as train_tr
from training.train_isolation import train as train_is


def main():
    print("=" * 60)
    print("          FIN.DR - Master Model Training pipeline")
    print("=" * 60)

    # 1. Train TensorFlow
    try:
        train_tf()
    except Exception as e:
        print(f"[Master] Error training TensorFlow model: {e}")

    # 2. Train PyTorch Transformer
    try:
        train_tr()
    except Exception as e:
        print(f"[Master] Error training PyTorch Transformer: {e}")

    # 3. Train Isolation Forest
    try:
        train_is()
    except Exception as e:
        print(f"[Master] Error training Isolation Forest: {e}")

    print("\n" + "=" * 60)
    print("  All training modules executed successfully!")
    print("  Checkpoints saved to models/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
