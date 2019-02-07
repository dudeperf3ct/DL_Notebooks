import face_recognition
from PIL import Image
import glob
import time
import os

start = time.time()
all_face_encoding = []
img_list = []
for img_file in glob.glob('known_imgs/*.jpg'):
	all_image = face_recognition.load_image_file(img_file)
	all_face_encoding.append(face_recognition.face_encodings(all_image)[0])
	img_list.append(os.path.basename(os.path.splitext(img_file)[0]))
end = time.time()
print (len(all_face_encoding))
print (all_face_encoding)
print (img_list)
print ("Time required", (end-start))