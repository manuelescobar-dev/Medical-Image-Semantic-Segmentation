# Preprocessing

## Installation
Openslide is required to read whole slide images. To install it, follow the [instructions](https://openslide.org/).
## Usage
`patching.py` contains the necessary methods for dividing whole slide images in smaller patches that can be processed by the semantic segmentation network.

**Parameters:**
- Patch Size: Resulting dimensions of each patch. For examplem 2000 produces 2000px x 2000px images.
- Level: Magnification level. The smaller the level, the more resolution is used, resulting in more patches.