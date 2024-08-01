import torch
print("CUDA available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")


# (perio2) E:\LT4\PERIO_24.03.05\WebPlatform>conda list
# # packages in environment at C:\Users\cwh92\anaconda3\envs\perio2:
# #
# # Name                    Version                   Build  Channel
# asgiref                   3.8.1                    pypi_0    pypi
# ca-certificates           2024.7.2             haa95532_0
# certifi                   2024.7.4                 pypi_0    pypi
# charset-normalizer        3.3.2                    pypi_0    pypi
# colorama                  0.4.6                    pypi_0    pypi
# contourpy                 1.2.1                    pypi_0    pypi
# cycler                    0.12.1                   pypi_0    pypi
# django                    4.2.1                    pypi_0    pypi
# filelock                  3.15.4                   pypi_0    pypi
# fonttools                 4.53.1                   pypi_0    pypi
# fsspec                    2024.6.1                 pypi_0    pypi
# idna                      3.7                      pypi_0    pypi
# importlib-resources       6.4.0                    pypi_0    pypi
# intel-openmp              2021.4.0                 pypi_0    pypi
# jinja2                    3.1.4                    pypi_0    pypi
# kiwisolver                1.4.5                    pypi_0    pypi
# markupsafe                2.1.5                    pypi_0    pypi
# matplotlib                3.9.1                    pypi_0    pypi
# mkl                       2021.4.0                 pypi_0    pypi
# mpmath                    1.3.0                    pypi_0    pypi
# networkx                  3.2.1                    pypi_0    pypi
# numpy                     1.26.4                   pypi_0    pypi
# opencv-python             4.10.0.84                pypi_0    pypi
# openssl                   3.0.14               h827c3e9_0
# packaging                 24.1                     pypi_0    pypi
# pandas                    2.2.2                    pypi_0    pypi
# pillow                    10.2.0                   pypi_0    pypi
# pip                       24.0             py39haa95532_0
# psutil                    6.0.0                    pypi_0    pypi
# py-cpuinfo                9.0.0                    pypi_0    pypi
# pyparsing                 3.1.2                    pypi_0    pypi
# python                    3.9.19               h1aa4202_1
# python-dateutil           2.9.0.post0              pypi_0    pypi
# pytz                      2024.1                   pypi_0    pypi
# pyyaml                    6.0.1                    pypi_0    pypi
# requests                  2.32.3                   pypi_0    pypi
# scipy                     1.13.1                   pypi_0    pypi
# seaborn                   0.13.2                   pypi_0    pypi
# setuptools                69.5.1           py39haa95532_0
# shapely                   2.0.3                    pypi_0    pypi
# six                       1.16.0                   pypi_0    pypi
# sqlite                    3.45.3               h2bbff1b_0
# sqlparse                  0.5.0                    pypi_0    pypi
# sympy                     1.13.0                   pypi_0    pypi
# tbb                       2021.13.0                pypi_0    pypi
# thop                      0.1.1-2209072238          pypi_0    pypi
# torch                     2.3.1+cu118              pypi_0    pypi
# torchaudio                2.3.1+cu118              pypi_0    pypi
# torchvision               0.18.1                   pypi_0    pypi
# tqdm                      4.66.4                   pypi_0    pypi
# typing-extensions         4.12.2                   pypi_0    pypi
# tzdata                    2024.1                   pypi_0    pypi
# ultralytics               8.1.23                   pypi_0    pypi
# urllib3                   2.2.2                    pypi_0    pypi
# vc                        14.2                 h2eaa2aa_4
# vs2015_runtime            14.29.30133          h43f2093_4
# wheel                     0.43.0           py39haa95532_0
# zipp                      3.19.2                   pypi_0    pypi