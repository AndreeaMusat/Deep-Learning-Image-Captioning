# Automated image captioning using deep learning

## Training a model
**Pre-training step** for downloading the ground truth captions, the images and CNN features for the Flickr8k dataset:
```
./make_flickr8k_setup.sh
```

Usage for **training an image captioning model for flickr8k**:
```
cd src
make
```

## The Model
The **flow** of the data:

![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/flow.png)

**Feature extraction:**\
For extracting the features from the images, Caffe was used. The choice is motivated by the fact that Caffe provides already trained state of the art CNNs that are easy to use and faster than other deep learning frameworks. Before feeding the images to a CNN, they have to be resized to a fixed size and the mean image of the dataset has to be subtracted, as unnormalized data do not produce the expected outputs. The features are extracted from one layer at the end of the network. After experiments with
two different CNN architectures, GoogleNet and VGG-16, GoogleNet was chosen, as it produced better captions. The 1000-dimensional features extracted with GoogleNet and downsampled to a space with less dimensions using a Dense layer (in order to reduce the amount of computations) are the input of the RNN at the first time step.

**Language model:**\
Before feeding the captions to the language generating model, several steps have to be followed:\
• The entire dataset is read. A vocabulary V is formed with all the words that have a frequency higher than a specifed threshold and each word is assigned an index between 0 and |V| - 1. Mappings from indices to words (idx2tok) and viceversa (tok2idx) are created for transforming the raw sentences to arrays of indices and for being able to create natural
language sentences from the sampled indices at the end.\
• Each caption is read. Some captions are much longer than all the others, so they are clipped to a certain length. \
• A special start token is inserted at the beginning of the sentence and a special end token is appended at the end of the sentence. \
• If the sentence has words that are not found in the vocabulary, they are replaced with an unknown token. Because the number of words is reduced, them dimenionality of the input is reduced, so memory and additional computation are saved. \
• The sentence is transformed to a vector of indices using the mapping from the first step. \
• Batches of fixed size of arrays of indices are fed to an embedding layer which is responsible for representing each token in a multidimensional feature space. \
The features are then fed into an RNN model that, at each time step, generates a probability distribution for the next word. \

**Captioning model architecture:**\
![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/captionmodel.png)

## Training details:
Multiple layers of RNN/LSTM/GRU can be stacked. The deeper the model is, the higher its capacity to learn, but also the number of parameters increases, so it is slower to train. The optimal number of layers in the experiments was 2. The batch size influences how good an approximation of the real gradient is the current gradient, computed on some part of the data only. Because of memory related considerations, the maximum batch size for experiments was 256 and it produced the best results. \
The optimizer used was Adam with the default parameters. \
Regarding the word embeddings, after some training epochs, the randomly initialized ones yield results comparable to the ones obtained with models that have pretrained embeddings. The optimal embedding size was found to be about 200, a greater number of features leading to overfitting and a smaller number of features leading to a model that is not capable of learning.
To produce a softer probability over the classes and result in more diversity, a softmax temperature of 1.1 was used. When the temperature is lower, the model tends to generate repetitive words and be more conservative in its samples. \

## Results
![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/results1.png)
![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/results2.png)
![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/results3.png)
![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/results4.png)
![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/results5.png)

**Scores on the test set for Flickr8K**
![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/bestscores.png)

## Comments
Two datasets were used for experiments: Flickr8K and Flickr30K. The linguistic data was collected using crowd-sourcing approaches (Amazon's Mechanical Turk) and each image was captioned by 5 different people, thus varying in quality, as some of the Turkers were not even proficient in English. The length of the collected captions depended on the background of the workers, the qualified ones producing longer sentences. As it can be seen, they are not very diverse. From a sample of 5 images from Flickr8k, 3 of them have dogs and the other 2 contain people doing sports, which is proof that the images are
not very diverse, so the captioning model overfits easily. Flickr30k, on the other hand, having a larger corpus, has more diverse images, which leads to lower evaluation scores.

![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/dataset_samples.png)

As a result of having multiple workers from Amazon's Mechanical Turk work on this task, the style in which the image is captioned might be different. Using the Universal Sentence Encoder as a similarity measure of the sentences, it can be observed that the captions can be quite different and even written in different styles. A heatmap with all the pairwise similarities between the 5 ground truth captions and the automatically generated caption shows that the 4 th caption diverges the most from the others, including the one generated by the model. At a closer look, it is noticed that the style used in the sentence is different, having a more story-like sound. Because of this, it is very diffcult to correctly
evaluate the results and also, it is very challenging to train a model on data that is not uniform.

![alt text](https://github.com/AndreeaMusat/Deep-Learning-Image-Captioning/blob/master/results/sentencesimilarity.png)
