import os

# image size tests
os.system("python3 cnn_d1.py 0 1 90")
os.system("python3 cnn_d1.py 0 2 90")
os.system("python3 cnn_d1.py 0 4 90")
os.system("python3 cnn_d1.py 0 8 90")

os.system("python3 cnn_d1.py 2 2 32")
os.system("python3 cnn_d1.py 2 2 64")
os.system("python3 cnn_d1.py 2 2 90")
os.system("python3 cnn_d1.py 2 2 128")
os.system("python3 cnn_d1.py 2 2 256")

