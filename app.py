from fastai.vision.all import *
import gradio as gr
def is_cat(x): return x[0].isupper()
learn1 = load_learner('model.pkl')
categories = ('Dog','Cat')

def classify_images(img):
    pred,idx,probs = learn1.predict(img)
    return dict(zip(categories, map(float,probs)))
image = gr.Image()
label = gr.Label()
intf = gr.Interface(fn=classify_images,inputs=image,outputs=label)
intf.launch(inline=False)