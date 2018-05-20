import os

os.system("python3 vgg16_model.py 1 90 0.2 0.2")
os.system("python3 vgg16_model.py 2 90 0.2 0.2")
os.system("python3 vgg16_model.py 3 90 0.2 0.2")

os.system("python3 vgg16_model.py 1 128 0.2 0.2")
os.system("python3 vgg16_model.py 2 128 0.2 0.2")
os.system("python3 vgg16_model.py 3 128 0.2 0.2")

os.system("python3 vgg16_model.py 1 90 0.4 0.4")
os.system("python3 vgg16_model.py 2 90 0.4 0.4")
os.system("python3 vgg16_model.py 3 90 0.4 0.4")

os.system("python3 vgg16_model.py 1 128 0.4 0.4")
os.system("python3 vgg16_model.py 2 128 0.4 0.4")
os.system("python3 vgg16_model.py 3 128 0.4 0.4")

os.system("python3.6 vg-low-data-low-aug.py")
os.system("python3.6 vg-low-data-high-aug.py")
os.system("python3.6 vg-high-data-low-aug.py")
os.system("python3.6 vg-high-data-high-aug.py")

# also running image tests on vgg16
os.system("python3.6 vgg16_64.py")
os.system("python3.6 vgg16_256.py")