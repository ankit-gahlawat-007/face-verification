# face-verification

This is a live face-verification program.

Given a picture it will provide the percentage match with faces on web-camera.

It uses Facenet to generate 512-dim vector embeddings. 
I have used keras-facenet package: https://pypi.org/project/keras-facenet/

------------------------------------------------------------------------------------------------------------------------------------

How to use?

It uses python3 and following packages:
- keras_facenet
- scipy
- cv2

Put a picture in the data directory. (Currently there is a photo of Brad Pitt in the data directory, you can use that to match how much he resembles you)
In the python script, change the value of variable 'input' with new picture name.
Run "python3 face-verification.py".

------------------------------------------------------------------------------------------------------------------------------------

Future add-ons?

Instead of just 1 picture, we can change the code to match from multiple photos in the data directory.

We can use say SVM classifier to classify different images into their names, so instead of just the match it will output the name of the person.
