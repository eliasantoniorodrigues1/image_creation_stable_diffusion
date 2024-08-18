from flask import Flask, render_template, request
from PIL import Image
import secrets
from diffusers import StableDiffusionPipeline
import torch
import cv2
from datetime import datetime
import re


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'Processing on {device}')

app = Flask(__name__)

pipeline = StableDiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-4").to(device)
pipeline.enable_freeu(b1= 1.3, b2= 1.4, s1= 0.9, s2= 0.2)

super_res = cv2.dnn_superres.DnnSuperResImpl_create()
super_res.readModel('EDSR_x4.pb')
super_res.setModel('edsr', 4)

# generate random secret key
app.config['SECRET_KEY'] = secrets.token_hex(16)

@app.route('/')
def hello():
    # home page
    
    return render_template(
        "index.html", 
        # pass variables into the HTML template
        btn_range = range(3), 
        prompt_images = ["/static/images/placeholder_image.png" for i in range(3)]
    )

@app.route('/prompt', methods=['POST', 'GET'])
def prompt():
    # generate images from user prompt
    print("user prompt received:", request.form['prompt_input'])

    for i in range(3):
      image = pipeline(request.form['prompt_input']).images[0]
      image.save(f'static/images/demo_img_{str(i)}.png')

    return render_template(
        "index.html", 
        # pass variables into the HTML template
        btn_range = range(3), 
        prompt_images = [f'static/images/demo_img_{str(i)}.png' 
                         for i in range(3)]
    )

@app.route('/supersample', methods=['POST', 'GET'])
def supersample():
    # unique img name
    img_id = re.sub(r'\D', '', str(datetime.today()))

    # enlarge and save prompt image in high quality
    print("save button", request.form['save_btn'], "was clicked!")

    demo_img = cv2.imread(f'./static/images/demo_img_{request.form["save_btn"]}.png')
    demo_img = cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB)

    XL_img = super_res.upsample(demo_img)
    XL_img = Image.fromarray(XL_img)
    XL_img.save(f'./static/images/saved/img_{img_id}.png')

    return render_template(
        "index.html", 
        # pass variables into the HTML template
        btn_range = range(3), 
        prompt_images = [f'static/images/demo_img_{str(i)}.png' 
                         for i in range(3)]
    )

if __name__ == '__main__':
    # run application
    app.run(
        host = '0.0.0.0', 
        port = 8000, 
        debug = True
    )   

