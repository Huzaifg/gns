from enum import Enum

# Enum for user to provide dtype
class DType(Enum):
  SINGLE = "single" # This is no AMP
  HALF = "half" # Recommended for inference
  MIXED = "mixed" # This is AMP