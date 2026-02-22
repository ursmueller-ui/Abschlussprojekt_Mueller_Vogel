import json 
import os
from datetime import datetime

FILE_PATH = "storage.json"
max_saves = 5

def load_all_models():
    #Lädt alle gespeicherten Modelle

    if not os.path.exists(FILE_PATH):
        return []

    try:
        with open(FILE_PATH, "r") as f:
            return json.load(f)
    # wenn Fehler beim Laden auftritt leeres Array zurückgeben
    except:
        return []

def save_all_models(models):

    with open(FILE_PATH, "w") as f:
        json.dump(models, f, indent=2)


# Modelle speichern

def save_model(
    name,
    width,
    height,
    nx,
    nz,
    res,
    e_mod,
    constraints,
    symmetry,
    target_mass,
    step
):
    #Es werden nur die Parameter gespeichert

    models = load_all_models()

    model = {

        "name": name,

        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),

        "width": width,
        "height": height,

        "nx": nx,
        "nz": nz,
        "res": res,

        "e_mod": e_mod,

        "constraints": constraints,

        "symmetry": symmetry,

        "target_mass": target_mass,
        "step": step
    }

    # Vorne einfügen
    models.insert(0, model)

    # Nur letzte 5 behalten
    models = models[:max_saves]

    save_all_models(models)


# Nur die Namen zurückgeben

def get_model_names():
    models = load_all_models()
    
    return [m["name"] for m in models]


#bestimmtes Modell laden
def load_model(index):

    models = load_all_models()

     # Nur etwas zurückgeben, wenn der Index gültig ist
    if index < len(models):
        return models[index]

    return None
