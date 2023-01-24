import os
import sys

def getListOfFiles(dirName, exts:tuple = ()):
    listOfFile = os.listdir(dirName)
    allFiles = list()

    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath, exts)
        else:
            if exts:
                if fullPath.endswith(exts):
                    allFiles.append(fullPath)
            else:
                allFiles.append(fullPath)
                
    return allFiles

def getUniqueFilename(file:str):
    result = file
    dir, filename = os.path.split(file)

    copy_id = 1
    while os.path.exists(result):
        name, ext = filename.split('.')
        result = os.path.join(dir, name + '_{0}.'.format(copy_id) + ext)
        copy_id += 1

    return result