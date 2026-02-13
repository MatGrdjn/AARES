import pandas as pd
import os
import json
from datetime import datetime

def save_parameters(model_name, parameters, rmse, mae, r2, path):
    """
    Enregistre les scores et les paramètres.
    """

    params_string = json.dumps(parameters)
    
    log_entry = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'r2': round(r2, 4),
        'parameters': params_string  # Une seule colonne pour tout
    }
    
    df_new = pd.DataFrame([log_entry])
    
    file_exists = os.path.isfile(path)
    df_new.to_csv(path, mode='a', index=False, header=not file_exists, encoding='utf-8')
    
    print(f"Résultats de {model_name} enregistrés dans {path}")