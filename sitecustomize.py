import os

# Work around duplicate OpenMP runtime initialization in this local conda env.
# This is only for local execution convenience; it does not change model logic.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
