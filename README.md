# Listed-Hashtag-Generator

I have constructed a ML model for generating hashtags when an user uploads an image in my Interface.For training the model,I have used tensorflow modules over a dataset of images and it's mapped captions(took from an open source available in Internet).From the captions,I am generating hashtags.For my UI,I have used Flask,css is my backend file.To start the process,I have to execute my gen_hashtags.py file once to train the model.Then for each time when the user uploads an image,hashtags function in gen_hashtags file gets called.
