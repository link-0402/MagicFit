# Magic Fit

Magic Fit is a Blender add-on built for common Final Fantasy XIV modding workflows. It features a purpose built, advanced weight transfer logic to minimize clipping and improve C+ compatibility. Besides transferring from a source, it supports generating clean, fitting weights for hairs, faces and skirts. 
The Line Up Tool allows easily lining up models from other games for fitting, while Resize lets you easily fit outfits to a target bod, both from another FFXIV base body as well as other bodies (from different games). Various fitting brushes and miscellaneous tools help with easier model editing and sculpting, circumventing common issues and frustrations.
Finally, a mesh repair mode can rebuilt bad geometry, build cleaner UVs (and easily transfer textures to it, close holes in the mesh and repair grid structures when textures become warped as a result of editing.

![The Magic Fit tab in Blender's sidebar](docs/images/sidebar.png)

**New here?** Install it below. The [wiki](https://github.com/link-0402/MagicFit/wiki) explains every tool step by step.

## Requirements

- **Blender 5.2 or newer.**
- **64-bit Windows for Weight Transfer.** It needs SciPy, and the copy that comes with the add-on is built for Windows. Everything else works on any system.
- **A full body mesh with an armature** (the skeleton it's attached to) for the tools that work against a body, such as weight transfer, line up and fitting. Lining up hair needs the face model's skin instead.
- **FFXIV installed on the same computer** for Face Weights. It reads the game's faces from your install the first time you use it.

## Installation

Add this repository to Blender once and install it, you'll then get future updates automatically.

1. In Blender, open **Edit → Preferences → Get Extensions**.
2. Click **Repositories**, click **+**, and choose **Add Remote Repository**.
3. Paste this address:

   ```
   https://raw.githubusercontent.com/link-0402/MagicFit/main/blender_repo/index.json
   ```
4. Tick **Check for Updates on Startup**.
5. Find **Magic Fit** in the list and click **Install**. The download is about 32 MB because it includes SciPy, which Blender sets up for you.
6. In the 3D Viewport, press **N** and open the **Magic Fit** tab.

More help is on the wiki's [Installation and Setup](https://github.com/link-0402/MagicFit/wiki/Installation-and-Setup) page.

## Guide

| I want to… | Use | Read |
| --- | --- | --- |
| Make clothing move properly with a character's body and minimize clipping | Weight Transfer, Connect Parts | [Weight Transfer](https://github.com/link-0402/MagicFit/wiki/Weight-Transfer) |
| Fix weights where they still look wrong | Weight Brushes | [Weight Brushes](https://github.com/link-0402/MagicFit/wiki/Weight-Brushes) |
| Give skirts proper skirt weights and prevent heeled shoes from bending | Skirt and Heel weights | [Skirt and Heel Weights](https://github.com/link-0402/MagicFit/wiki/Skirt-and-Heel-Weights) |
| Apply weights to hair or a custom face without a direct weight transfer source | Hair and Face weights | [Hair Weights](https://github.com/link-0402/MagicFit/wiki/Hair-Weights), [Face Weights](https://github.com/link-0402/MagicFit/wiki/Face-Weights) |
| Fit clothing around a body | Body Fit, Fit Move | [Body Fit](https://github.com/link-0402/MagicFit/wiki/Body-Fit) |
| Move an outfit to another body mod or size | Auto Resize | [Auto Resize](https://github.com/link-0402/MagicFit/wiki/Auto-Resize) |
| Adjust an imported model from another game for easy porting | Line Up | [Line Up](https://github.com/link-0402/MagicFit/wiki/Line-Up) |
| Fix stretched textures, torn seams, messy meshes, bad shading or UVs | Mesh Repair | [Texture Relax and Close Rips](https://github.com/link-0402/MagicFit/wiki/Texture-Relax-and-Close-Rips), [Rebuild Mesh](https://github.com/link-0402/MagicFit/wiki/Rebuild-Mesh) |
| See how a weighted model behaves with my Customize+ template applied | Customize+ | [Customize+](https://github.com/link-0402/MagicFit/wiki/Customize-Plus) |

## Help

Something not working? Try [Troubleshooting and FAQ](https://github.com/link-0402/MagicFit/wiki/Troubleshooting-and-FAQ). If that doesn't help, or a model from another game doesn't line up, please [open an issue](https://github.com/link-0402/MagicFit/issues).
You can additionally get support on the [Discord](https://discord.gg/w5vzsPXsE7).

## Credits

- **Weight Transfer** started as [Robust Weight Transfer](https://jinxxy.com/SentFromSpaceVR/robust-weight-transfer) by [sentfromspacevr](https://github.com/sentfromspacevr), which implements [Robust Skin Weights Transfer via Weight Inpainting](https://github.com/rin-23/RobustSkinWeightsTransferCode) by Rinat Abdrashitov et al.
- **Customize+** reproduces the template math of [Customize+](https://github.com/Aether-Tools/CustomizePlus) by Aether-Tools.
- **Bundled libraries:** [SciPy](https://scipy.org), [robust-laplacian](https://github.com/nmwsharp/robust-laplacian-py) and [XIVPy](https://github.com/Arrenval/XIVPy).

Papers, game data and licenses are on the wiki's [Credits](https://github.com/link-0402/MagicFit/wiki/Credits) page.

## License

Magic Fit is licensed under the [GNU General Public License v3.0 or later](magic_fit/LICENSE).
