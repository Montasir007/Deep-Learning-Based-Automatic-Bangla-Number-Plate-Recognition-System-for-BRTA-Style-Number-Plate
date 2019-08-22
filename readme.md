# to use this file open terminal and type:
python np.py --image number_plate.jpg



#this file support argparse so by chnaging image name it can be used for testing different image.


#in np.py I used 3 models for boosting accurecy


#"Dhame" model for recognizing dhaka and metro. In future this model could be extended for other city name


# "Ko" for reconginizing ko, kho, go or gho. further it could me modified for other characters too.


#"Model" for recognizing digits in number plate


#working procedure:


#first get the image name by command argument


#read the image file 


#then used the function imop (image operation) to split character from number plate


#first i splited the number plate into 2 horizontal segment. because these two part have to go through different model.


# i developed many model because number of data for numeric and nonnumeric character was not same. we get good amount of data for numeric. 
but non numeric data was very few. so i dont want to mix up good model with bad model. the numeric model is very strong in acuracy. 


# from upper horizontal part I just read 1st(Dhaka) and last(ko, kho, go or gho) character for prediction. as metro is same for all numberplate


#thn i used resize and fill funciton to resize input images.


# after get all character I passed them to model for prediction. and print the result into matplotlib images


# note, the image showed in matplotlibs. Its title is our result.



Now let me talk about functions used in this file:


1. imop: imop used for character segmentaion. I get the idea of this funciton from pyimagesearch.com. 
I modified it as our requirement. and splited images into part for boosting up accuracy in character segmentation. 


2. resize and fill1: this funciton is also used to convert images into same size(35x28) while trainig neural network for character recognition.
 As it was used for training data so its logical to convert test images into same size for boosting accuracy. 
this function convert images into 35x28 if some image lower than this size then it fill the image with average color of this image

3. 
resize and fill2: As size of "Dhaka"/"metro" and other digits are not same. so we need another resize and fill function. 


#after doing whole process i use os.remove to remove all the segmented images which was needed to write for prediction.    
