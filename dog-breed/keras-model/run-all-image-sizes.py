import os

# image size tests

os.system("python3 cnn_d1.py 0 1 32")
os.system("python3 cnn_d1.py 0 1 64")
os.system("python3 cnn_d1.py 0 1 90")
os.system("python3 cnn_d1.py 0 1 128")
os.system("python3 cnn_d1.py 0 1 256")

os.system("python3 cnn_d1.py 0 1 90")
os.system("python3 cnn_d1.py 0 2 90")
os.system("python3 cnn_d1.py 0 4 90")
os.system("python3 cnn_d1.py 0 8 90")

