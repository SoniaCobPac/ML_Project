## **Generating Music with Machine Learning**

### **Author:**
Sonia Cobo Pacios
The Bridge 2021 - The Bridge Digital Talent


---------

## **Project description:**

Music is associated with emotions, experiences and creativity, all o them considered human's qualities. 

Though this project doesn't have a hypothesis per se it has been done to prove that technology has advanced to the point where artificial intelligence, that cannot experience these feelings, can generate music.

This project will explain and show how music has been generated using differente types of neural networks in Python. For this aim different types of neural networks have been analised. Two Long short-term memory (LSTM) models (a Bidirectional and a Unidirectional network) and a Generative adversarial network (GAN) considering different layers and parameters.


---------


## **Project structure:**

### Pre-processing: 
Relevant information is extracted from MIDI files and codificated for its used as network input

### Modeling:
Five models were review for the project:
    - Unidirectional Long Short-Term Memory
    - Unidirectional Long Short-Term Memory with different parameters and layers
    - Bidirectional Long Short-Term Memory
    - GAN (LSTM)
    - GAN (LeakyRelu)

The notebook 'main' explains these processing in more detail.

### Post-processing:
The predicted output was decodified and saved back into a MIDI format.


---------


## **Project conclusion:**

This one-week project has proved that deep learning is capable of generating new melodies. Though results are far from perfect it is still impressive what a shallow neural network and a simple GAN network are capable to do. 

The study concludes that artificial intelligence can generate new melodies. Now it is only required to invest more time and research to improve the selected models to improve the predictions. There are differente ideas, all explained in the below section "next steps" that can be carried out to improve the study. 


---------


## **Next Steps:**

Different alternatives and additional work can always be done to improve this work. Due to the available time and resources it was not possible to implement all ideas. Some of them that can contribute to the improvement of the predictions are listed below:

- Change time signature to 4/4 to achieve more consistancy to the input and ease patterns
- Keep the length of all pieces the same
- Try more parameters and layers to all networks. However, deeper networks also required a more powerful computer.
- Play around with the number of epoch and batches to optimised results
- Train the model with files consisting of different instruments
- Clean the data based on sound frequency
- Convert MIDI files to WAV files automatically in Python


---------

