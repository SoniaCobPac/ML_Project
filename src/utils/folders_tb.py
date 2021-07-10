import json
import os

def read_json(path, filename):
    """
    Read a json and return a object created from it.
    Args:
        file path and file name
    
    Returns: json object.
    """
    try:
        with open(path + filename , "r+", encoding="latin-1") as outfile:
            json_readed = json.load(outfile)
        return json_readed
    except Exception as error:
        raise ValueError(error)