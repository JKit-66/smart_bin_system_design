"""from roboflow import Roboflow
rf = Roboflow(api_key="wJYMHjporLYxtMW02sxw")
project = rf.workspace("uni-project-1").project("project-1-thfya")
dataset = project.version(1).download("yolov8")
"""

from ultralytics import YOLO
import ultralytics.hub.utils as hub_utils
import matplotlib
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")

#matplotlib.use('TkAgg')

hub_utils.ONLINE = False

# Load the model.
model = YOLO('/mainfs/scratch/jkl1a20/newfolder/roboflow/TrainData/best.pt')
 
# Training.
results = model.train(
   data='/mainfs/scratch/jkl1a20/newfolder/roboflow/TrainData/data.yaml',
   imgsz=800,  
   epochs=20,
   name='finale',
   plots=True,
   resume=True
)


now = datetime.now()
current_time = now.strftime("%H:%M:%S")