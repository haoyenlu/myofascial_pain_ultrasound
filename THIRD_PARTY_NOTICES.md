# Third-party notices

This project is licensed under Apache-2.0 (see `LICENSE`). It includes
modified copies of the following MIT-licensed components. Each adapted file
has a header pointing back to its source.

| Component | Upstream | Files in this repo |
|---|---|---|
| denoising-diffusion-pytorch | https://github.com/lucidrains/denoising-diffusion-pytorch | `models/diffusion_classifier/denoising_diffusion_pytorch/` |
| video-classification-3d-cnn-pytorch | https://github.com/kenshohara/video-classification-3d-cnn-pytorch | `models/resnet.py` |
| 3D-ResNets-PyTorch | https://github.com/kenshohara/3D-ResNets-PyTorch | `models/diffusion_classifier/resnet_3d.py` |
| vidaug | https://github.com/okankop/vidaug | `vidaug/augmentors/` (except `sampler.py`, which is original) |

## Pretrained weights downloaded at runtime (not redistributed)

- **Kinetics-pretrained 3D ResNet-50** (`pretrained/resnet-50-kinetics.pth`),
  from the video-classification-3d-cnn-pytorch project. You download it
  yourself; see the upstream repository for terms.
- **`MCG-NJU/videomae-base`**, fetched from the Hugging Face Hub by
  `videomae/`. These weights are released by their authors under
  **CC-BY-NC-4.0** (non-commercial). Check that license before using the
  VideoMAE baseline commercially.

---

## denoising-diffusion-pytorch

```
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## video-classification-3d-cnn-pytorch and 3D-ResNets-PyTorch

```
MIT License

Copyright (c) 2017 Kensho Hara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## vidaug

```
MIT License

Copyright (c) 2018 okankop

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
