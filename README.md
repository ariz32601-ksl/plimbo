# PLIMBO (Modernized Staging Layer)

Planarian Interface for Modelling Body Organization. This repository serves as a modernized, high-velocity staging fork of the legacy PLIMBO planarian engine.

## Modernizations Included
1. **SciPy Environment Patch:** Completely replaced the deprecated and removed `scipy.misc.imresize` image processing hooks with a native, high-performance PIL (Pillow) vector matrix wrapper compatible with modern Python 3.11+ layers.
2. **Sovereign Engine Mapping:** Relinked core simulation pipelines directly to the accelerated `sovereign_betse` hardware matrix layer for Apple Silicon (Metal Performance Shaders).
3. **Dependency Stabilization:** Standardized configuration tracking with modern `scikit-learn` and serialization imports.

