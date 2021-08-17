# radcap_project
This is code for a project in medical school trying to automate descriptions of medical images. 
It is based on different implementations of the CNN-RNN design. Each implementation has one python file for the model, one file to prepare the data in batches, one file to train the network and one file to generate captions with the trained network. 
There is also code in the radcap_project/data_preprocessing/ folder for creating the vocabulary. The file radcap_bodypartsplit_data.json, which should by in the preproccessing folder, contains specific information for my dataset. Use your own datafile there instead and chance the code to read your data.

To train the network run the train file.

To generate captions run the caption-generation file, see below. 

Implementation of the CNN-RNN from the paper "Show and tell" https://arxiv.org/abs/1411.4555

-File 1(the model): radcap_project/cnn_rnn_models/cnn_rnn_vinalys/

-File 2(dataprep): radcap_project/cnn_rnn_models/cnn_rnn_vinalys/dataHandler.py

-File 3(train): radcap_project/train.py

-File 4(generate captions): radcap_project/sample.py