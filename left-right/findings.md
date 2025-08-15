This is a summary of the findings of the experiments performed by Alejandro (me) for left-right classification.
This can also be found in google docs here: https://docs.google.com/document/d/1H8xpN85DDk6kFHwQyag9mjbpfw4XDsGBx3glGnHmncw/edit?tab=t.0 

## Dataset
Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6190745/pdf/sdata2018211.pdf
Download Dataset: https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698 

The dataset I trained on, which I called the “yuriy” dataset, is paradigm #4 in the paper from the first link above. Participants pressed either the “D” or “L” keys on a keyboard while their EEG brain signals were recorded. They pressed “D” with their left and “L” with their right hands, making it ideal for our use case. In all the results, I used data from 1 single subject (subject C of the database) over 2 sessions. The sample frequency is 200 Hz. 
You can download the dataset in the second link above.
The data fetching code is in the “fetch_yuriy_data.py” file in get_yuriy_data(samples_around: int). When fetching the data, our script gets all the data from samples_around number of samples around the key press. I didn’t do any preprocessing to the data, except that the ridge model scales it using RobustScaler.

## Models
In total, I trained 5 models. The accuracies shown are the best test accuracies I could get while fine-tuning the model and training.

### Ridge Model: 77% best accuracy
The Ridge model I trained is RidgeClassifierCV from sklearn, which makes a prediction based on a single timestamp of EEG data. The best timestamp is determined by cross-validating across all possible timestamps and picking the one with the best accuracy. This approach is based on the Meta paper.
Code: the training file is ridge_yuriy.ipynb
Results: we got a maximum of 77% accuracy, and the best timestamp was around 0.24 seconds after the keypress.

### ShallowConvNet (96%, ~34K params) and EEGNetv4 (95%, ~2K params)
ShallowConvNet is a model from this research paper: https://arxiv.org/pdf/1703.05051. It uses a smart non-linear transformation that’s based on properties of EEG data.
EEGNet is from this paper: https://arxiv.org/pdf/1611.08024. It uses a depthwise separable convolution to drastically reduce the number of parameters, making it a very lightweight option for high performance.
I preloaded both untrained models from braindecode and trained them.
Code: training code can be found in preexisting_cnn_training.ipynb. The trained models are saved in the preexisting_shallow_model.pth and preexisting_eegnet_model.pth files.

### ConvLUN Model (97%, 270K params)
This model is based on this paper: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2020.00338/full. It uses a very simple convolution, normalization, activation, pooling, and repeat model, with flatten and linear layers at the end.
Code: I wrote the model out in Pytorch in the file conv_lun_model.py. I can’t guarantee I replicated it exactly. The training code is in conv_lun_training.ipynb.

### 1D Model (91%, 335K params)
This model is different from the other CNNs, since this model uses 1-dimensional hidden layers. It does 1 temporal convolution (across each channel independently), 1 spatial convolution (across each timestamp independently), and then 3 move 1-dimensional convolutions.
I did this as an experiment, since the CNN in the paper by Défossez (which is the paper referenced by the Meta paper when talking about the convolutional module) used 1-dimensional hidden layers. I found this model harder to work with, and it produced worse results, while having the most parameters.
Model Code: cnn_1d_model.py
Training Code: cnn_1d_training.ipynb

## Training Code
To train the LUN and 1D models, I wrote the models out in Pytorch and trained them using Pytorch Lightning. 
NeuralNetTrainer (in the neural_net_trainer.py file): This is a lightning module I built using Pytorch Lightning that helps train the neural networks. It takes the model, optimizer, params, etc, and trains the model while tracking the training, testing, and validation losses. I did this mostly to learn Pytorch Lightning for the future, but this trainer is reusable to train different neural networks and can come in handy moving forward.
MetricsLogger (in training_utils.py): This Pytorch Lightning Callback tracks the training and validation metrics to display and analyze after training.


## Findings
It’s clear that left-right classification is very achievable with CNNs and even with a very simple ridge model. Even with no preprocessing and forgetting to scale the data, we had very good performance. 
The best working model was the Lun CNN, but it also had a lot of parameters (270K). ShallowConvNet got close performance with only ~34K, and EEGNet is even more lightweight with ~2K. We can learn from the techniques used in these 3 models to create our own CNNs. For example, I really like the idea of using a depthwise-separable convolution to reduce the parameter size employed by EEGNet.
These models will be useful when testing our data. Since we know they work, even in rough form, we should be able to get similar results on our data. Additionally, the training scripts will be helpful when training CNNs in the future.
