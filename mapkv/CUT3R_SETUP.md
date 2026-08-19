# CUT3R provider setup

MapKV uses the official CUT3R repository as a separate geometry process. No
CUT3R source or checkpoint is vendored into this repository.

- Upstream: <https://github.com/CUT3R/CUT3R>
- Tested commit: `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`
- Tested checkpoint: `cut3r_512_dpt_4_64.pth`
- Exchange contract: generated PNG prefix in, normalized NPZ/JSON pointmaps,
  confidences, c2w poses, and intrinsics out.

Keep its environment separate from InSpatio:

```bash
git clone https://github.com/CUT3R/CUT3R third_party/CUT3R
git -C third_party/CUT3R checkout 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
python -m venv --system-site-packages third_party/mapkv_cut3r_env
third_party/mapkv_cut3r_env/bin/pip install accelerate gdown viser
```

Follow CUT3R's own installation instructions for its remaining dependencies
and CUDA RoPE extension. With PyTorch 2.7, its CUDA kernel may need the narrow
compatibility edit `tokens.type()` to `tokens.scalar_type()` before compiling.
The checkpoint loader also needs `weights_only=False` to retain pre-2.6
behavior because the trusted official checkpoint contains its OmegaConf model
constructor. These are external local compatibility patches and are not copied
into MapKV.

CUT3R and its weights retain their upstream copyright and license terms. This
adapter is intended for the research prototype described here; check upstream
terms before redistribution or commercial use.

The fixed-pose incremental alignment flow was adapted from the research
reference runjiali-rl/vmem@39291e4f272f6b4f270691d930926ab5930f942e,
especially extern/CUT3R/surfel_inference.py. VMem is MIT-licensed. MapKV does
not vendor VMem source; the adapter-side previous-depth gradient mask is an
independent compatibility implementation for the upstream CUT3R optimizer.
