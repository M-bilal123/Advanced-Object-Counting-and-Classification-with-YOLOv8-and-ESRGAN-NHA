import cv2
import time
import logging
import torch
import numpy as np
from PIL import Image
import RRDBNet_arch as arch
# Disable all logging
logging.disable(logging.CRITICAL)  # This disables all logging levels

# Disable print statements by redirecting stdout to devnull
import os
import sys
# sys.stdout = open(os.devnull, 'w')

from ultralytics import solutions
from ultralytics import YOLO

from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

from PIL import Image

class ObjectCounter(BaseSolution):


    def __init__(self, **kwargs):
       
        super().__init__(**kwargs)

        self.in_count = 0 
        self.out_count = 0  
        self.counted_ids = []  
        self.classwise_counts = {}  
        self.region_initialized = False  

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.classifier = YOLO('classifier_weights/best.pt')  # Load your custom classifier model
        self.detailed_counts = {} 
        self.countt=0

        # Initialize the super-resolution model
        self.sr_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.sr_model.load_state_dict(torch.load('SR_Model/RRDB_ESRGAN_x4.pth'), strict=True)
        self.sr_model.eval()
        self.sr_model = self.sr_model.to(torch.device('cuda'))

    def count_objects(self, track_line, box, track_id, prev_position, cls):
        """
        Counts objects within a polygonal or linear region based on their tracks.

        Args:
            track_line (Dict): Last 30 frame track record for the object.
            box (List[float]): Bounding box coordinates [x1, y1, x2, y2] for the specific track in the current frame.
            track_id (int): Unique identifier for the tracked object.
            prev_position (Tuple[float, float]): Last frame position coordinates (x, y) of the track.
            cls (int): Class index for classwise count updates.

        Examples:
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id = 1
            >>> prev_position = (120, 220)
            >>> cls = 0
            >>> counter.count_objects(track_line, box, track_id, prev_position, cls)
        """
        if prev_position is None or track_id in self.counted_ids:
            return

        centroid = self.r_s.centroid
        dx = (box[0] - prev_position[0]) * (centroid.x - prev_position[0])
        dy = (box[1] - prev_position[1]) * (centroid.y - prev_position[1])

        if len(self.region) >= 3 and self.r_s.contains(self.Point(track_line[-1])):
            self.counted_ids.append(track_id)
            # For polygon region
            if dx > 0:
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1

        elif len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):
            self.counted_ids.append(track_id)
            
            # Extract the object image
            x1, y1, x2, y2 = map(int, box)
            obj_img = self.current_frame[y1:y2, x1:x2]
            
            if obj_img.size == 0:
                print("Warning: Empty object image for classification.")
                return
            
            try:
                # Convert numpy array to PIL Image first
                cv2.imshow('obj_img', obj_img)
                obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
                obj_img = Image.fromarray(obj_img)
                
                # Convert to tensor and move to GPU
                obj_img_tensor = torch.from_numpy(np.array(obj_img)).float().div(255).permute(2, 0, 1).unsqueeze(0).to(torch.device('cuda'))
                
                # Apply super-resolution
                with torch.no_grad():
                    sr_img_tensor = self.sr_model(obj_img_tensor)
                
                # Convert back to numpy array
                sr_img = sr_img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
                sr_img = (sr_img * 255).clip(0, 255).astype(np.uint8)
                
                # Convert back to BGR for OpenCV
                sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                
                # Save and display the super-resolved image
                cv2.imwrite(f'out/sr_obj_img{self.countt}.png', sr_img)
                self.countt += 1
                # cv2.imshow('sr_obj_img', sr_img)
                
                # Pass the super-resolved image to the classifier
                results = self.classifier(sr_img, verbose=False)
                
                if results and results[0].probs:
                    detailed_class = results[0].probs.top1  # Get the predicted class
                    detailed_label = results[0].names[detailed_class]  # Get the class name
                    print('#####################',detailed_class,detailed_label,'###############')
                    
                    # Store detailed classification
                    if detailed_label not in self.detailed_counts:
                        self.detailed_counts[detailed_label] = {"IN": 0, "OUT": 0}
                    
                    # Update counts based on direction
                    if dx > 0 and dy > 0:
                        self.in_count += 1
                        self.classwise_counts[self.names[cls]]["IN"] += 1
                        self.detailed_counts[detailed_label]["IN"] += 1
                    else:
                        self.out_count += 1
                        self.classwise_counts[self.names[cls]]["OUT"] += 1
                        self.detailed_counts[detailed_label]["OUT"] += 1
                else:
                    print("Warning: No classification results.")
                    
            except Exception as e:
                print(f"Classification error: {e}")
                # Fall back to original counting if classification fails
                if dx > 0 and dy > 0:
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                else:
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1

    def store_classwise_counts(self, cls):
        """
        Initialize class-wise counts for a specific object class if not already present.

        Args:
            cls (int): Class index for classwise count updates.

        This method ensures that the 'classwise_counts' dictionary contains an entry for the specified class,
        initializing 'IN' and 'OUT' counts to zero if the class is not already present.

        Examples:
            >>> counter = ObjectCounter()
            >>> counter.store_classwise_counts(0)  # Initialize counts for class index 0
            >>> print(counter.classwise_counts)
            {'person': {'IN': 0, 'OUT': 0}}
        """
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        """
        Displays object counts on the input image or frame.

        Args:
            im0 (numpy.ndarray): The input image or frame to display counts on.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
        """
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def count(self, im0):
        """
        Processes input data (frames or object tracks) and updates object counts.

        This method initializes the counting region, extracts tracks, draws bounding boxes and regions, updates
        object counts, and displays the results on the input image.

        Args:
            im0 (numpy.ndarray): The input image or frame to be processed.

        Returns:
            (numpy.ndarray): The processed image with annotations and count information.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> processed_frame = counter.count(frame)
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        # Store current frame
        self.current_frame = im0.copy()
        
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)  # Extract tracks

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # Draw region

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Draw bounding box and counting region
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            self.store_tracking_history(track_id, box)  # Store track history
            self.store_classwise_counts(cls)  # store classwise counts in dict

            # Draw tracks of objects
            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(cls), True), track_thickness=self.line_width
            )

            # store previous position of track for object counting
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_line, box, track_id, prev_position, cls)  # Perform object counting

        self.display_counts(im0)  # Display the counts on the frame
        self.display_output(im0)  # display output with base class function

        return im0  # return output image for more usage

cap = cv2.VideoCapture("main.asf")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(f"Frame width: {w}, frame height: {h}, fps: {fps}")
vehicle_classes = ['car', 'truck', 'bus', 'van','motorcycle']  
# Define region points
region_points = [(232, 635), (660, 435)]  # For line counting
# region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)] 
# region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]  

# Video writer
# Resize dimensions to 720p
output_width = 1280
output_height = 720
video_writer = cv2.VideoWriter("object_counting_output-test.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))

# Init Object Counter
counter = ObjectCounter(
    show=True,  
    region=region_points,  
    model="yolo11m.pt",  
    classes=[2,5,7],  
    show_in=True,  
    show_out=True,  
    line_width=2,
    verbose=False  
)

# Process video
prev_time = time.time()
frame_count = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    # Resize frame to 720p
    im0 = cv2.resize(im0, (output_width, output_height))
    im0 = counter.count(im0)
    video_writer.write(im0)
    
    # Calculate and log FPS
    frame_count += 1
    if frame_count % 30 == 0:  # Update FPS every 30 frames
        current_time = time.time()
        fps = 30 / (current_time - prev_time)
        print(f"Processing FPS: {fps:.2f}")
        prev_time = current_time
print('#####################',counter.detailed_counts,'###############')
cap.release()
video_writer.release()
cv2.destroyAllWindows()
