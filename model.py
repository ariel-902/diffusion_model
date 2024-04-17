import gdown

# problem model 1
url = "https://drive.google.com/u/1/uc?id=14kxZhjtr1-9Cl4JzcW_2MH-KS3D3glXw&export=download"
output = "./p1_model.pt"
gdown.download(url, output)

# problem model 3
# svhn extractor
url = "https://drive.google.com/u/1/uc?id=1o0EZAJht6QPEc9Riza07h1URqZJf3mLr&export=download"
output = "./p3_svhn_extractor.ckpt"
gdown.download(url, output)

# svhn predictor
url = "https://drive.google.com/u/1/uc?id=1Hi2pfpP9L5ZyfkkUj_0KJzfBl7A1MUZQ&export=download"
output = "./p3_svhn_predictor.ckpt"
gdown.download(url, output)

# usps extractor
url = "https://drive.google.com/u/1/uc?id=1Rkm66igigzEZZ1IgPl9sLouSHnjXESQx&export=download"
output = "./p3_usps_extractor.ckpt"
gdown.download(url, output)

# usps predictor
url = "https://drive.google.com/u/1/uc?id=1ZzhwYIsCfMkHTSr07jkVEN_JITpeoYWf&export=download"
output = "./p3_usps_predictor.ckpt"
gdown.download(url, output)