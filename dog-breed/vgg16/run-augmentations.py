import os

os.system("python3.6 vg-low-data-low-aug.py")
os.system("python3.6 vg-low-data-high-aug.py")
os.system("python3.6 vg-high-data-low-aug.py")
os.system("python3.6 vg-high-data-high-aug.py")
 # also running image tests on vgg16
os.system("python3.6 vgg16_64.py")
os.system("python3.6 vgg16_256.py")
