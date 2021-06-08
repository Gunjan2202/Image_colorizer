from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from skimage.io import imsave
from skimage.transform import resize
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from PIL import Image,ImageChops
import logging 
import cv2

app = Flask(__name__)
app.secret_key = "hello"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
model = load_model('unet_final_ver1O1.hdf5')
model2 = load_model('alpha_final.hdf5')
# model3 = load_model('unet_final_ver1O1.hdf5')
UPLOAD_FOLDER = './files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
files = [f for f in os.listdir('.') if os.path.isfile(f)]
checkInception = False
for f in files:
    if f == "inception.h5":
        checkInception = True
        inception = load_model('inception.h5', compile=False)
        break
if not checkInception:
    inception = InceptionResNetV2(weights='imagenet', include_top=True)
    inception.save('inception.h5')


def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    embed = inception.predict(grayscaled_rgb_resized)
    return embed


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    index_page = 'index.html'
    if request.method == 'POST':
        try:
            url = request.form['url']
            if 'examples' in url:
                color_file = process(url)
                color_file2 = process2(url)
                # color_file3 = process3(url)
                return render_template(index_page,res='static/examples/girl.jpg')
        # check if the post request has the file part
        except:
            logging.exception('')
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            color_file = process(file.filename)
            color_file2 = process2(file.filename)
            # color_file3 = process3(file.filename)
            return render_template(index_page, og=color_file[0], res=color_file[1],res2=color_file2[1])


    return render_template(index_page)

# def process(img):
#     if 'examples' in img:
#         im = Image.open(img)
#         name = img.split('.')[0].split('/')[-1]
        
#     else:
#         im = Image.open('files/' + img)
#         name = img.split('.')[0]
        
#     old_size = im.size  # old_size[0] is in (width, height) format
#     ratio = float(256)/max(old_size)
#     new_size = tuple([int(x*ratio) for x in old_size])
#     im = im.resize(new_size, Image.ANTIALIAS)
#     new_im = Image.new("RGB", (256, 256))
#     new_im.paste(im, ((256-new_size[0])//2,(256-new_size[1])//2))
#     new_im.save('static/processed_png/' + name + ".png","PNG")
#     a = np.array(img_to_array(load_img('static/processed_png/' + name +'.png')))
#     a = a.reshape(1,256,256,3)
#     color_me_embed = create_inception_embedding(a)
#     a = rgb2lab(1.0/255*a)[:,:,:,0]
#     a = a.reshape(a.shape+(1,))
#     output = model.predict([a, color_me_embed])
 
#     output = output * 128
    
#     for i in range(len(output)):
#         cur = np.zeros((256, 256, 3))
#         cur[:,:,0] = a[i][:,:,0]
#         cur[:,:,1:] = output[i]
#         imsave(f'static/colored_img/{name}.png',(lab2rgb(cur)))
#         trim(Image.open(f'static/processed_png/{name}.png')).save(f'static/processed_png/{name}.png')
#         trim(Image.open(f'static/colored_img/{name}.png')).save(f'static/colored_img/{name}.png')
#         return (f'static/processed_png/{name}.png',f'static/colored_img/{name}.png')

     

# def process2(img):
#     print("///////////////////////////////////")
#     print(img)
#     print("////////////////////////////////////")
#     if 'examples' in img:
#         im = Image.open(img)
#         name = img.split('.')[0].split('/')[-1]
#         name2=name+"22"
#     else:
#         im = Image.open('files/' + img)
#         name = img.split('.')[0]
#         name2=name+"22"
#     old_size = im.size  # old_size[0] is in (width, height) format
#     ratio = float(256)/max(old_size)
#     new_size = tuple([int(x*ratio) for x in old_size])
#     im = im.resize(new_size, Image.ANTIALIAS)
#     new_im = Image.new("RGB", (256, 256))
#     new_im.paste(im, ((256-new_size[0])//2,(256-new_size[1])//2))
#     new_im.save('static/processed_png/' + name + ".png","PNG")
#     a = np.array(img_to_array(load_img('static/processed_png/' + name +'.png')))
#     a = a.reshape(1,256,256,3)
#     color_me_embed = create_inception_embedding(a)
#     a = rgb2lab(1.0/255*a)[:,:,:,0]
#     a = a.reshape(a.shape+(1,))
    
#     output2 = model2.predict([a, color_me_embed])

#     output2 = output2 * 128
#     for i in range(len(output2)):
#         cur = np.zeros((256, 256, 3))
#         cur[:,:,0] = a[i][:,:,0]
#         cur[:,:,1:] = output2[i]
#         imsave(f'static/colored_img/{name2}.png',(lab2rgb(cur)))
#         trim(Image.open(f'static/processed_png/{name}.png')).save(f'static/processed_png/{name}.png')
#         trim(Image.open(f'static/colored_img/{name2}.png')).save(f'static/colored_img/{name2}.png')
#         return (f'static/processed_png/{name}.png',f'static/colored_img/{name2}.png')

# def process3(img):
#     if 'examples' in img:
#         im = Image.open(img)
#         name = img.split('.')[0].split('/')[-1]
#         name3=name+"33"
#     else:
#         im = Image.open('files/' + img)
#         name = img.split('.')[0]
#         name3=name+"33"
#     old_size = im.size  # old_size[0] is in (width, height) format
#     ratio = float(256)/max(old_size)
#     new_size = tuple([int(x*ratio) for x in old_size])
#     im = im.resize(new_size, Image.ANTIALIAS)
#     new_im = Image.new("RGB", (256, 256))
#     new_im.paste(im, ((256-new_size[0])//2,(256-new_size[1])//2))
#     new_im.save('static/processed_png/' + name + ".png","PNG")
#     a = np.array(img_to_array(load_img('static/processed_png/' + name +'.png')))
#     a = a.reshape(1,256,256,3)
#     color_me_embed = create_inception_embedding(a)
#     a = rgb2lab(1.0/255*a)[:,:,:,0]
#     a = a.reshape(a.shape+(1,))
    
#     output3 = model3.predict([a, color_me_embed])

#     output3 = output3 * 128
#     for i in range(len(output3)):
#         cur = np.zeros((256, 256, 3))
#         cur[:,:,0] = a[i][:,:,0]
#         cur[:,:,1:] = output3[i]
#         imsave(f'static/colored_img/{name3}.png',(lab2rgb(cur)))
#         trim(Image.open(f'static/processed_png/{name}.png')).save(f'static/processed_png/{name}.png')
#         trim(Image.open(f'static/colored_img/{name3}.png')).save(f'static/colored_img/{name3}.png')
#         return (f'static/processed_png/{name}.png',f'static/colored_img/{name3}.png')

def process(img):
    if 'examples' in img:
        im = Image.open(img)
        name = img.split('.')[0].split('/')[-1]
        # name2=name+"11"
    else:
        im = Image.open('files/' + img)
        name = img.split('.')[0]
        # name2=name+"11"
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(256)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])


    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (256, 256))
    new_im.paste(im, ((256-new_size[0])//2,(256-new_size[1])//2))
    new_im.save('static/processed_png/' + name + ".jpeg","JPEG")

    f1='static/processed_png/' + name +'.jpeg'
    #f1= "./dataset/testing/149.jpg"
    a=cv2.imread(f1,0) 
    a=a[:144,:144] 
    a=np.array(a);
    #a = np.array(img_to_array(load_img('static/processed_png/' + name +'.png')))
    a = a.reshape(1,144,144,1)
    a=a/255. 
    # color_me_embed = create_inception_embedding(a)
    # a = rgb2lab(1.0/255*a)[:,:,:,0]
    # a = a.reshape(a.shape+(1,))

    #output3 = model3.predict([a, color_me_embed])
    output=model.predict(a)
    mx=np.amax(output[0])
    mn=np.amin(output[0])
    # img=output[0]
    output[0]=output[0]-mn
    output[0]=output[0]/(mx-mn);
    #output3 = output3 * 128

    for i in range(len(output)):
        cur = np.zeros((144, 144, 3))
        # cur[:,:,0] = a[i][:,:,0]
        cur[:,:,:] = output[i]
        imsave(f'static/colored_img/{name}.jpeg',(cur))
        Image.open(f'static/processed_png/{name}.jpeg').save(f'static/processed_png/{name}.jpeg')
        Image.open(f'static/colored_img/{name}.jpeg').save(f'static/colored_img/{name}.jpeg')
        return (f'static/processed_png/{name}.jpeg',f'static/colored_img/{name}.jpeg')

def process2(img):
    if 'examples' in img:
        im = Image.open(img)
        name = img.split('.')[0].split('/')[-1]
        name2=name+"22"
    else:
        im = Image.open('files/' + img)
        name = img.split('.')[0]
        name2=name+"22"
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(256)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])


    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (256, 256))
    new_im.paste(im, ((256-new_size[0])//2,(256-new_size[1])//2))
    new_im.save('static/processed_png/' + name + ".jpeg","JPEG")

    f1='static/processed_png/' + name +'.jpeg'
    #f1= "./dataset/testing/149.jpg"
    a=cv2.imread(f1,0) 
    a=a[:144,:144] 
    a=np.array(a);
    #a = np.array(img_to_array(load_img('static/processed_png/' + name +'.png')))
    a = a.reshape(1,144,144,1)
    a=a/255. 
    # color_me_embed = create_inception_embedding(a)
    # a = rgb2lab(1.0/255*a)[:,:,:,0]
    # a = a.reshape(a.shape+(1,))

    #output3 = model3.predict([a, color_me_embed])
    output2=model2.predict(a)
    mx=np.amax(output2[0])
    mn=np.amin(output2[0])
    # img=output[0]
    output2[0]=output2[0]-mn
    output2[0]=output2[0]/(mx-mn);
    #output3 = output3 * 128

    for i in range(len(output2)):
        cur = np.zeros((144, 144, 3))
        # cur[:,:,0] = a[i][:,:,0]
        cur[:,:,:] = output2[i]
        imsave(f'static/colored_img/{name2}.jpeg',(output2[0]))
        Image.open(f'static/processed_png/{name}.jpeg').save(f'static/processed_png/{name}.jpeg')
        # Image.open(f'static/colored_img/{name2}.jpeg').save(f'static/colored_img/{name2}.jpeg')
        return (f'static/processed_png/{name}.jpeg',f'static/colored_img/{name2}.jpeg')




def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

if __name__ == "__main__":
    app.run(debug=True)
