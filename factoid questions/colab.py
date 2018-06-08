# Install the PyDrive wrapper & import libraries.
# This only needs to be done once per notebook.
import keras
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import os

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def seeFiles():
  file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
  for file1 in file_list:
    print('title: %s, id: %s' % (file1['title'], file1['id']))
    
def downloadFile(Id, name):
  file_id = Id
  downloaded = drive.CreateFile({'id': file_id})
  downloaded.GetContentFile(name)
  
def createFile(name):
  file5 = drive.CreateFile()
# Read file and set it as a content of this instance.
  file5.SetContentFile(name)
  file5.Upload() # Upload the file.
  
def downloadFiles():
  downloadFile('1yxyKCeDmJuHqiCYIogI2bj19_sI_D6Rc', "BioASQ-trainingDataset5b.json")