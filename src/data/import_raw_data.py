import requests
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    """import filenames from bucket_folder_url in raw_data_relative_path"""
    if os.path.exists(raw_data_relative_path)==False:
        os.makedirs(raw_data_relative_path)
    for filename in filenames :
        input_file = os.path.join(bucket_folder_url,filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        object_url = input_file
        print(f'downloading {input_file} as {os.path.basename(output_file)}')
        response = requests.get(object_url)
        if response.status_code == 200:
            content = response.text
            text_file = open(output_file, "wb")
            text_file.write(content.encode('utf-8'))
            text_file.close()
        else:
            print(f'Error accessing the object {input_file}:', response.status_code)
        return output_file
                
def split_data(df):
    target = df['silica_concentrate']
    features = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
               
def main(raw_data_relative_path="./data/raw", 
        filenames = ["raw.csv"],
        bucket_folder_url= "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/"          
        ):
    """ Upload data from AWS s3 in ./data/raw
    """
    output_filepath='./data/processed'
    
    output_file=import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)

    df = pd.read_csv(output_file, sep=",")   
      
    X_train, X_test, y_train, y_test = split_data(df)

    if os.path.exists(output_filepath)==False:
        os.makedirs(output_filepath)

    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        filepath = os.path.join(output_filepath, f'{filename}.csv')
        file.to_csv(filepath, index=False)
    
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()