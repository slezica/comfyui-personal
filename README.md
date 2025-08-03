# ComfyUI Personal Nodes

A custom node collection for ComfyUI containing simplified workflow nodes and enhanced UI features for my personal use.

## Installation

1. Clone in ComfyUI `custom_nodes` directory
2. Install required dependencies
3. Restart ComfyUI

The nodes will appear under the "personal" category in the node browser.

## Nodes

- **UseCheckpoint** - Load model checkpoint with simplified interface
- **UseLora** - Apply LoRA with streamlined controls
- **UseControlNet** - ControlNet integration with auto-type detection
- **UseIPAdapter** - IP-Adapter implementation with plus model support
- **UseStyleModel** - Style transfer using CLIP vision models
- **UseImage** - Image processing with inpainting and differential diffusion
- **UseInstantID** - Face identity preservation with InstantID
- **UseInfiniteYou** - Advanced face pose and embedding extraction
- **GenerateImage** - Unified sampling with multiple scheduler support (TCD, DPM++, Euler)
- **UpscaleImage** - Image upscaling with model selection
- **OwlDetector** - Object detection using OWL-ViT models

## UI Enhancements

- **Enhanced Search** - Fuzzy search with personal nodes prioritized
- **Keyboard Shortcuts**:
  - `Tab` - Centered node search dialog
  - `Shift + Arrow Keys` - Align selected nodes
  - `Shift + L` - Arrange nodes horizontally
  - `Shift + C` - Arrange nodes vertically
  - `Shift + W/H` - Resize nodes to match selection
  - `Escape` - Close dialogs
- **Node Management** - Copy/paste widget values between nodes
- **Auto-configuration** - Smart parameter adjustment for different samplers

## Dependencies

Requires other ComfyUI custom node packages:

- [ComfyUI_InstantID](https://github.com/cubiq/ComfyUI_InstantID)
- [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
- [ComfyUI_InfiniteYou](https://github.com/bytedance/ComfyUI_InfiniteYou)
- [ComfyUI-TCD](https://github.com/JettHu/ComfyUI-TCD)
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
