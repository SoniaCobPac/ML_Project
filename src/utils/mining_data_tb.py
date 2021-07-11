# data manipulation
import numpy as np
import pandas as pd 
from sqlalchemy import create_engine

# manipulate midi files
import glob
from music21 import *
#from music21 import converter, instrument, note, chord, meter, stream, duration, corpus
import pygame

# route files
import os
import sys

# ml model
import pickle
import tensorflow as tf

# EDA

sep = os.sep

def route (steps, option=1):
    """
    This function appends the route of the file to the sys path
    to be able to import files from/to other foders.

    Param: Steps (int) to go up to the required folder
    """
    if option == 1:
        route = os.getcwd()
    elif option == 2:
        route = os.path.dirname(os.path.abspath(__file__))

    for i in range(steps):
        route = os.path.dirname(route)
    sys.path.append(route)
    return route
    

def info_midi(path, filename):
    """
    It returns all midi file information given its path and filename.

    """
    # Convert to Score object
    file = converter.parse(path + filename)
    components = []
    # read file information
    for element in file.recurse():  
        components.append(element)
    return components


def transpose_key(path, path_1, name):
    """
    This function converts MIDI file of any kety to C major or A minor key.

    Params: Path of the original MIDI file and path where the converted file is to be saved.
    """
    import music21

    # major conversions
    majors = dict([("A-", 4),("G#", 4),("A", 3),("A#", 2),("B-", 2),("B", 1),("C", 0),("C#", -1),("D-", -1),("D", -2),("D#", -3),("E-", -3),("E", -4),("F", -5),("F#", 6),("G-", 6),("G", 5)])
    minors = dict([("G#", 1), ("A-", 1),("A", 0),("A#", -1),("B-", -1),("B", -2),("C", -3),("C#", -4),("D-", -4),("D", -5),("D#", 6),("E-", 6),("E", 5),("F", 4),("F#", 3),("G-", 3),("G", 2)])       

    # os.chdir("./")
    for file in glob.glob(path + f"{name}.mid"):
        score = music21.converter.parse(file)
        key = score.analyze('key')
        
        # print key.tonic.name, key.mode
        if key.mode == "major":
            halfSteps = majors[key.tonic.name]
            
        elif key.mode == "minor":
            halfSteps = minors[key.tonic.name]
        
        newscore = score.transpose(halfSteps)
        key = newscore.analyze("key")

        #print(key.tonic.name, key.mode)
        newFileName = f"{name}.mid"
        newscore.write("midi", path_1 + newFileName)


def get_notes_per_song(path, filename, save_path, save_name):
    """
    This function extracts all the notes, rests and chords from one midi file
    and saves it in a list in the converted_data folder.

    Param: Path of the midi file, its filename (str), path where the list will be saved and its name.
    """
    components = info_midi(path, filename)
    note_list = []
    
    for element in components:
        # note pitches are extracted
        if isinstance(element, note.Note):
            note_list.append(str(element.pitch))
        # chords are extracted
        elif isinstance(element, chord.Chord):
            note_list.append(".".join(str(n) for n in element.normalOrder))    
        # rests are extracted
        elif isinstance(element, note.Rest):
            note_list.append("REST")   

    with open(save_path + save_name, "wb") as filepath:
        pickle.dump(note_list, filepath)
    
    return note_list


# Not in use if the create_dataframe function is used
def get_all_notes(path, save_name, save_path):
    """
    This function extracts all the notes, rests and chords from all midi files 
    and saves it in a list in the converted_data folder.

    Param: Path of the midi file, path where the list will be saved and its name.
    """
    all_notes = []
    list_path = os.listdir(path)
    for filename in list_path:
        output = get_notes_per_song(path, filename, save_path, save_name)
        all_notes += output
        
    return all_notes


def save_elem (save_path, save_name, list_to_save):
    """
    This function saves a list for its later use.
    Param: Path where the list will be saved, its name and the list itself.
    """
    with open(save_path + save_name, "wb") as filepath:
        pickle.dump(list_to_save, filepath)


def load_notes (path, filename):
    """
    Load the note list containing pitches, rests and chords.
    
    Param: Path of the saved note list, and its name as string
    """
    with open(path + filename, "rb") as f:
        loaded_notes = pickle.load(f)
        return loaded_notes


def create_dataframe(path, save_path, save_name):
    """
    Create a dataframe and a list with all pieces title and the notes, rests and chrods included in the piece. 

    Param: Path of the midi file, filename (str), path where the list will be saved and its name.
    """

    list_path = os.listdir(path)
    piece_list = []
    notes = []
    # extracting all songs from the list
    for elem in list_path:
        output = get_notes_per_song(path, elem, save_path, save_name)
        piece_list.append(elem[:-4])
        notes.append(output)

    # create dataframe
    df = pd.DataFrame.from_dict({"Piece":piece_list, "Notes":notes}, orient="index")
    df = df.transpose()

    return df


def save_dataframe(path, dataframe, dataframe_name):
    """
    Create a cvs file from a dataframe.
    Param: Path to save the dataframe, dataframe to save and its name as string
    """
    dataframe.reset_index(inplace = True)
    if "index" in dataframe.columns:
        dataframe.drop(columns = "index", inplace = True)
    dataframe.to_csv(path + dataframe_name + ".csv", index=False)
    return "Your file has been saved"


def read_dataframe(path, filename):
    """
    Read dataframes from its path, and filename as string.
    """
    try:
        filename_df = pd.read_csv(path + filename + ".csv", sep=",")  
        return filename_df
    except:
        print("No file with such name has been found")


# PREPROCESSING

def prepare_sequences(pitchnames, notes, min_note_occurence, sequence_length, step):
    """ 
    This function creates the input and output sequences used by the neural network.
    It returns the x and y of the model.

    Param: 
        Pitchnames: List containing all notes, rests and chords
        Note: List of the input piece containing notes, rests and chords
        Sequence_length: Lenght of notes given to the model to help predict the next
        Step: Step (int) between one input sequence and the next one
    """
 
    # Calculate occurence
    note_occ = {}
    for elem in notes:
        note_occ[elem] = note_occ.get(elem, 0) + 1

    ignored_notes = set()
    for k, v in note_occ.items():
        if note_occ[k] < min_note_occurence:
            ignored_notes.add(k)
    
    # create a dictionary to convert pitches (strings) to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))  # rests are included  

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, step): 
        # remove ignored notes from the note list   
        if len(set(notes[i: i+ sequence_length + 1]).intersection(ignored_notes)) == 0:
            network_input.append(notes[i:i + sequence_length])
            network_output.append(notes[i + sequence_length])
    # array of zeros
    x = np.zeros((len(network_input), sequence_length, len(pitchnames)))
    y = np.zeros((len(network_input), len(pitchnames)))
    # exchange note values for their integer-code
    for i, sequence in enumerate(network_input):
        try:
            for j, note in enumerate(sequence):
                x[i, j, note_to_int[note]] = 1
            y[i, note_to_int[network_output[i]]] = 1
        except:
            continue

    return x, y

# Generate notes function is slightly different for GAN models
def prepare_sequences_gan(pitchnames, notes, min_note_occurence, sequence_length, step):
    """ 
    This function creates the input and output sequences used by the neural network.
    It returns the x and y of the model.

    Param: 
        Pitchnames: List containing all notes, rests and chords
        Note: List of the provided piece containing all notes, rests and chords
        Sequence_length: Lenght of notes given to the model to help predict the next
        Step: Step (int) between one input sequence and the next one
    """
     # Calculate occurence
    note_occ = {}
    for elem in notes:
        note_occ[elem] = note_occ.get(elem, 0) + 1

    ignored_notes = set()
    for k, v in note_occ.items():
        if note_occ[k] < min_note_occurence:
            ignored_notes.add(k)

    # create a dictionary to convert pitches (strings) to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))  # rests are included  

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - 2*sequence_length, step):    
        network_input.append(notes[i:i + sequence_length])
        network_output.append(notes[i + sequence_length : i + 2*sequence_length])

    x = np.zeros((len(network_input), sequence_length, len(pitchnames)))
    y = np.zeros((len(network_input), sequence_length, len(pitchnames)))
    for i, sequence in enumerate(network_input):
        try:
            for j, note in enumerate(sequence):
                x[i, j, note_to_int[note]] = 1
                y[i, j, note_to_int[network_output[i][j]]] = 1
        except:
            continue

    return x, y



# POST PROCESSING

def sample(preds, temperature=1.0):
    """
    Helper function to sample an index from a probability array

    """
    # probability distribution 
    preds = np.asarray(preds).astype("float64")
    # convert to the correct numpy array type
    preds = np.log(preds) / temperature
    # scaling
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_notes(pitchnames, notes, model, sequence_length, temperature=1.0):
    """ 
    Generate notes from the neural network based on a sequence of notes

    Param: 
        Piece: List containing all notes
        Notes: List of the provided piece containing notes
        Model: Neural network
        Temperature: int to control prediction randomness 

    """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(notes)-sequence_length-1)

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames)) 
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    pattern = notes[start: (start + sequence_length)] 
    prediction_output = []
    patterns = []

    # generate 500 notes, roughly two minutes of music
    for note_index in range(100):
        prediction_input = np.zeros((1, sequence_length, len(pitchnames)))
        for j, note in enumerate(pattern):
            prediction_input[0, j, note_to_int[note]] = 1.0
        preds = model.predict(prediction_input, verbose=0)[0] 
        
        next_index = sample(preds, temperature=temperature)
        next_note = int_to_note[next_index]

        pattern = pattern[1:]
        pattern.append(next_note)

        prediction_output.append(next_note)

        patterns.append(next_index)
  
    return prediction_output, patterns

# Generate notes function is slightly different for GAN models
def generate_notes_gan(pitchnames, notes, model, sequence_length, temperature=1.0):
    """ 
    Generate notes from the GAN network based on a sequence of notes 
    """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(notes)-sequence_length-1)

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames)) 
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    
    pattern = notes[start: (start + sequence_length)] 
    prediction_output = []
    patterns = []

    # generate 500 notes, roughly two minutes of music

    prediction_input = np.zeros((1, sequence_length, len(pitchnames)))
    for j, note in enumerate(pattern):
        prediction_input[0, j, note_to_int[note]] = 1.0
    preds = model.predict(prediction_input, verbose=0)[0]  

    for elem in list(preds):
        next_index = sample(elem, temperature=temperature)
        next_note = int_to_note[next_index]
    
        prediction_output.append(next_note)

        patterns.append(next_index)

    return prediction_output, patterns

# GAN
def generate_real_samples(x, n_samples,sequence_length):
    """
    Prepare random real training samples
    Params:
        x: Network input
        n_samples: Number of samples
    """
    # choose random instances
    start = np.random.randint(0, len(x)-sequence_length-1)
    # retrieve selected images
    x_real = x[start: (start+n_samples)] 
    # generate 'real' class labels (1)
    y_real = np.ones((n_samples, 1))

    return x_real, y_real


def generate_latent_points(x, n_samples):
    """
    Create random input for the generator model
    Params:
        x: Network input
        n_samples: Number of samples
    """
    import random
    # create random matrix of numbers 
    x_latent = np.zeros((n_samples, x.shape[1], x.shape[2]))
  
    for j, elem in enumerate(x_latent):
        for k, row in enumerate(elem):
            num = random.randint(0, x.shape[2])
            for i in range(len(row)):
                if i == num:
                    x_latent[j][k][i] = 1

    return x_latent


def generate_fake_data(x, g_model, n_samples):
    """
    Prepare fake samples
    Params:
        x: Network input
        g_model: Generator model
        n_samples: Number of samples
    """
    # create 'fake' class labels (0)
    y_fake = np.zeros((n_samples, 1))

    # generate points in latent space
    x_latent = generate_latent_points(x, n_samples)
    # predict outputs
    x_fake = g_model.predict(x_latent, verbose=0)

    return x_fake, y_fake

    
def gen_midi(prediction_output, path, filename):
    """ 
    This functions converts model predictions to midi files
    
    Param: Decodified prediction output (list) from the model and path to save the midi file.    
    """
    
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a rest
        elif ("REST" in pattern):
            new_rest = note.Rest(pattern)
            output_notes.append(new_rest)
        # pattern is a note
        else:
            new_note = note.Note(pattern)   
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    # save generated midi file
    midi_stream.write("midi", fp= path + filename)   # first output 01/07/2021

    return midi_stream


def play_music(path_filename):
    """
    Play music given a midi file path
    """
    #import music21
    try:
        # allow to stop the piece 
        pygame.mixer.init()
        clock = pygame.time.Clock() 
        pygame.mixer.music.load(path_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            # check if playback has finished
            clock.tick(10)

        freq = 44100    # audio CD quality
        bitsize = -16   # unsigned 16 bit
        channels = 2    # 1 is mono, 2 is stereo
        buffer = 1024    # number of samples
        pygame.mixer.init(freq, bitsize, channels, buffer)

    except KeyboardInterrupt:
        while True:
            action = input('Enter Q to Quit, Enter to Skip.').lower()
            if action == 'q':
                pygame.mixer.music.fadeout(1000)
                pygame.mixer.music.stop()
            else:
                break


def process_and_execute(p1, p2, p3, p4, p5, p6, name, output_name):
    """
    This functions is the combination of all pre-processing, modeling and post-procssing functions used in the project 
    so given a MIDI file this one function will return its prediction and play it in Jupyter.
    
    """
    transpose_key(p3, p1, name)
    note_list = get_notes_per_song(p3, f"{name}.mid", p6, save_name="pred_list")
    print(len(note_list))
    #x, y = md.prepare_sequences(pitchnames=pitchnames, notes=note_list, min_note_occurence=1, sequence_length=70, step=3)
    model = tf.keras.models.load_model(p2 + "baseline_lstm_1epoch_1song_2.h5")
    prediction_output, patterns = generate_notes(p4, notes=note_list, model=model, temperature=1.0, sequence_length=70)
    gen_midi(prediction_output, p5, f"{output_name}.mid")
    play_music(p5 + f"{output_name}.mid")


# To check errors
if __name__ == "__main__":
    sep = os.sep
    # path to raw data
    path = route(2, option=2) + sep + "data" + sep + "raw_data" + sep
    # path to data in the right key
    path_1 = route(2, option=2) + sep + "data" + sep + "converted_data" + sep
    # path to compiled notes list
    path_2 = route(2, option=2) + sep + "data" + sep + "notes" + sep
    # path to generated models
    path_3 = route(2, option=2) + sep + "models" + sep
    # path to generated midi files
    path_4 = route(2, option=2) + sep + "reports" + sep
    # path to output information
    path_5 = route(2, option=2) + sep + "data" + sep + "output" + sep
    # path to other songs
    path_6 = route(2, option=2) + sep + "data" + sep + "more_inst_data" + sep
    # path to resources
    path_7 = route(2, option=2) + sep + "resources" + sep

    pitchnames = load_notes(path_2, "pitchnames")

    def process_and_execute(name, output_name):
        """
        This functions is the combination of all pre-processing, modeling and post-procssing functions used in the project 
        so given a MIDI file this one function will return its prediction and play it in Jupyter.

        Params: Name of the input MIDI file and name of the predicted MIDI file so it will be saved.

        """
        transpose_key(path=path_6, path_1=path_1, name=name)
        note_list = get_notes_per_song(path=path_6, filename=f"{name}.mid", save_path=path_2, save_name="pred_list")
        print(len(note_list))
        #x, y = md.prepare_sequences(pitchnames=pitchnames, notes=note_list, min_note_occurence=1, sequence_length=70, step=3)
        model = tf.keras.models.load_model(path_3 + "baseline_lstm_1epoch_1song_2.h5")
        prediction_output, patterns = generate_notes(pitchnames=pitchnames, notes=note_list, model=model, temperature=1.0, sequence_length=100)
        gen_midi(prediction_output, path=path_4, filename=f"{output_name}.mid")
        play_music(path_4 + f"{output_name}.mid")

    process_and_execute(name="Naruto - sadness and sorrow", output_name="Naruto")
