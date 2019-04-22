To run the FID score calculator:

1. Be sure that the folder ./fid/GAN/fake and ./fid/VAE/fake are present
2. Run score_fid.py with the following arguments :
	- "./fid/GAN/" --model="./svhn_classifier_old.pt" for the GAN FID
	- "./fid/VAE/" --model="./svhn_classifier_old.pt" for the VAE FID
	==> use svhn_classifier_old.pt only if you run on an older version of pytorch.