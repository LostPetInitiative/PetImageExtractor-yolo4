import numpy as np
import base64

def npArrayToBase64str(npArray):
    npArrayBytes = npArray.tobytes()
    return base64.encodebytes(npArrayBytes).decode("utf-8").replace("\n","")

def Base64strToNpArray(b64Str:str):
    bytesObj = base64.decodebytes(b64Str.encode("utf-8"))
    npArray = np.frombuffer(bytesObj, dtype=np.float32)
    return npArray
