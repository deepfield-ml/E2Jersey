# run in terminal: pip install roboflow
  
from roboflow import Roboflow
rf = Roboflow(api_key="unauthorized")
project = rf.workspace("flashxyz").project("jerseydetection")
version = project.version(7)
dataset = version.download("coco")
                
