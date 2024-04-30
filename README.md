# Latent Intrinsics

## Environment
This packages requires the following libraries:
```
pytorch=2.0
diffusers=0.14.0
```
## Running code
Firstly, download and unzip the [MIT-relighting dataset](https://projects.csail.mit.edu/illumination/). Then modify the data path to your data folder at `main.py L57`

To run the code
```
python main.py
```
You can replace the default hugging face key for loading the pertained model with your personal key at `diffusion_extractor.py Line128 `

There are three pieces of the code:
- precompute diffusion features
- Train the model
- visualization of swaping the learned latent factors.

Once the features are precomputed and saved, you can set `precompute_feat_list = False` to prevent re-computation next time.
