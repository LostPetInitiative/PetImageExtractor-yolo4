from skimage import io
import tempfile
import os
import base64
import shutil



def imagesNpToStrList(npImages):
    tempDir = tempfile.mkdtemp()
    try:
        idx1 = 0
        images = []
        # encoding images
        for npImage in npImages:
            photoPath = os.path.join(tempDir,"{0}.jpeg".format(idx1))
            io.imsave(photoPath, npImage)
            #print("image {0} saved".format(photoPath))
            with open(photoPath, 'rb') as photoFile:
                photo = photoFile.read()
                #print("image {0} read".format(photoPath))
                image = {
                    'type': "jpg",
                    'data': base64.encodebytes(photo).decode("utf-8").replace("\n","")
                }
                images.append(image)
            idx1 += 1
        return images
    finally:
        shutil.rmtree(tempDir)

def imagesFieldToNp(images):
    tempDir = tempfile.mkdtemp()
    try:
        imgIdx = 0
        imagesNp = []
        # decoding images
        for image in images:
            imgType = image['type']
            image_b64 : str = image['data']
            imageData = base64.decodebytes(image_b64.encode("utf-8"))
            imageFilePath = os.path.join(tempDir,"{0}.{1}".format(imgIdx,imgType))
            with open(imageFilePath, "wb") as file1:             
                file1.write(imageData)
            try:
                if imageFilePath.endswith(".png"):
                    imNumpy = io.imread(imageFilePath,plugin='imread')
                else:
                    imNumpy = io.imread(imageFilePath)
                imagesNp.append(imNumpy)
            except Exception as exc1:
                print("Error calulating hash for one of the images ({0})".format(exc1))        
            imgIdx += 1
        return imagesNp
    finally:
        shutil.rmtree(tempDir)
