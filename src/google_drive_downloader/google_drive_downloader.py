"""
    Author: Andrea Palazzi

    Copyright (c) 2017 Andrea Palazzi
    
    https://github.com/ndrplz/google-drive-downloader/blob/master/google_drive_downloader/google_drive_downloader.py
"""

from __future__ import print_function
import requests
import zipfile
import warnings
from sys import stdout
from os import makedirs
from os.path import dirname
from os.path import exists


class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, unzip=False):
        """
        Downloads a shared file from google drive into a given folder.
        Optionally unzips it.
        Parameters
        ----------
        file_id: str
            the file identifier.
            You can obtain it from the sherable link.
        dest_path: str
            the destination where to save the downloaded file.
            Must be a path (for example: './downloaded_file.txt')
        overwrite: bool
            optional, if True forces re-download and overwrite.
        unzip: bool
            optional, if True unzips a file.
            If the file is not a zip file, ignores it.
        Returns
        -------
        None
        """

        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(dest_path) or overwrite:

            session = requests.Session()

            print('Downloading {} into {}... '.format(file_id, dest_path), end='')
            stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            GoogleDriveDownloader._save_response_content(response, dest_path)
            print('Done.')

            if unzip:
                try:
                    print('Unzipping...', end='')
                    stdout.flush()
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        z.extractall(destination_directory)
                    print('Done.')
                except zipfile.BadZipfile:
                    warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination):
        with open(destination, "wb") as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)