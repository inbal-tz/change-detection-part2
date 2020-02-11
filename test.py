from PIL import Image

filename = r'D:\Users Data\inbal.tlGIP\Desktop\part b\labels\train/00027_13.jpg'
filename1 = r'D:\Users Data\inbal.tlGIP\Desktop\part b\labels\train/00019_7.jpg'
filename2 = r'D:\Users Data\inbal.tlGIP\Desktop\part b\labels\train/00000_4.jpg'
label = Image.open(filename2).convert('P')

print(list(label.getdata()))
