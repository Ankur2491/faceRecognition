import face_recognition
from PIL import Image, ImageDraw

image_of_bill = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]


image_of_steve = face_recognition.load_image_file('./img/known/Steve Jobs.jpg')
steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]

image_of_elon = face_recognition.load_image_file('./img/known/Elon Musk.jpg')
elon_face_encoding = face_recognition.face_encodings(image_of_elon)[0]

image_of_ankur = face_recognition.load_image_file('./img/known/Ankur Sharma.jpg')
ankur_face_encoding = face_recognition.face_encodings(image_of_ankur)[0]

image_of_vivek = face_recognition.load_image_file('./img/known/Vivek Singh.jpg')
vivek_face_encoding = face_recognition.face_encodings(image_of_vivek)[0]

image_of_nitish = face_recognition.load_image_file('./img/known/Nitish Roy.jpg')
nitish_face_encoding = face_recognition.face_encodings(image_of_nitish)[0]

image_of_abdul = face_recognition.load_image_file('./img/known/Abdul.jpg')
abdul_face_encoding = face_recognition.face_encodings(image_of_abdul)[0]

image_of_rupen = face_recognition.load_image_file('./img/known/Rupen Raj.jpg')
rupen_face_encoding = face_recognition.face_encodings(image_of_rupen)[0]

known_face_encodings = [
  bill_face_encoding,
  steve_face_encoding,
  elon_face_encoding,
  ankur_face_encoding,
  vivek_face_encoding,
  nitish_face_encoding,
  abdul_face_encoding,
  rupen_face_encoding
]

known_face_names = [
  "Bill Gates",
  "Steve Jobs",
  "Elon Musk",
  "Ankur Sharma",
  "Vivek Singh",
  "Nitish Roy",
  "Abdul Kadir Khan",
  "Rupen Raj"
]

test_image = face_recognition.load_image_file('./img/groups/group.jpg')

face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)
#print(f'Face Locations {face_locations}, Face Encodings {face_encodings}')

pil_image = Image.fromarray(test_image)

draw = ImageDraw.Draw(pil_image)

for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)

    name = "Unknown Person"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    draw.rectangle(((left, top),(right,bottom)), outline=(0,0,0))
    
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height - 10),(right, bottom)), fill=(0,0,0), outline=(0,0,0))
    draw.text((left+6, bottom-text_height-5), name, fill=(255,255,255,255))

del draw

pil_image.show()
