import Main
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import threading


sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util
def tensor():
    #t1.start()
    #t2.start()
    MODEL_NAME = 'inference_graph'

    
    CWD_PATH = os.getcwd()

    
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

    
    NUM_CLASSES = 2

    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)


    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    
    video = cv2.VideoCapture(1)
    ret = video.set(3,1280)
    ret = video.set(4,720)

    while(True):

        
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        cv2.imshow('Object detector', ret)

        #Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        cv2.imshow('Object detector', frame)
        

        
        if cv2.waitKey(1) == ord('q'):
            
            break

        # All the results have been drawn on the frame, so it's time to display it.
        #try:
            #if not os.path.exists('XXXX'):
                
                #os.makedirs('XXXX')
        #except OSError:
                
            #print ('Error: Creating directory of data')

        #currentFrame = 0
        #while(True):
            
            #for i in range(0,1000):
            
                #ret, frame = video.read()
                
                
                #name = './XXXX/frame' + str(currentFrame) + '.jpg'
                #print ('Creating...' + name)
                #cv2.imwrite(name, frame)
                
                
                #currentFrame += 1
        
                #if(currentFrame == 100):

                    #t2.start()
                    #time.sleep(10)
                
            
          
        #break
            
        

    
    video.release()
    cv2.destroyAllWindows()


t1 = threading.Thread(target = tensor)
t2 = threading.Thread(target = Main.main)

t1.start()

t1.join()
t2.join()





