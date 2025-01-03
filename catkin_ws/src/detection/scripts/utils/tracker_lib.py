import numpy as np
import cv2
import threading

DEEP_SORT = False
if DEEP_SORT:
    from deep_sort_realtime.deepsort_tracker import DeepSort
else:
    from scripts.utils.sort.sort import Sort

class TrackedObject(object):
    tracker_cls = cv2.legacy.TrackerKCF_create
    STS_VISIBLE = 0
    STS_LOST = -1
    def __init__(self, id, label, confidence):
        super().__init__()
        self.id = str(id)
        self.label = str(label)
        self.confidence = int(confidence)
        self.status = self.STS_LOST
        self.tracker_lock = threading.Lock()
        self.last_bbox = None

    def set_bbox(self, frame, bbox):
        x0,y0,x1,y1 = bbox
        with self.tracker_lock:
            self.tracker = self.tracker_cls()
            self.tracker.init(frame, [x0,y0,x1-x0,y1-y0])
            self.last_bbox = [x0,y0,x1-x0,y1-y0]
        self.status = self.STS_VISIBLE

    def get_bbox(self,frame):
        if not hasattr(self,'tracker'): return self.last_bbox
        if self.status == self.STS_LOST: return self.last_bbox

        with self.tracker_lock:
            ret, bbox = self.tracker.update(frame)
        if not ret:
            self.status = self.STS_LOST
            return self.last_bbox
        x,y,w,h = bbox
        self.last_bbox = [int(x),int(y),int(x+w),int(y+h)]
        return self.last_bbox

class Tracker(object):
    def __init__(self):
        self.__instances = dict()
        self.instances_lock = threading.Lock()
        if DEEP_SORT:
            self.id_tracker = DeepSort(max_age=5,
                                n_init=2,
                                nms_max_overlap=1.0,
                                max_cosine_distance=0.3,
                                nn_budget=None,
                                override_track_class=None,
                                embedder="mobilenet",
                                half=True,
                                bgr=True,
                                embedder_gpu=None,
                                embedder_wts=None,
                                polygon=False,
                                today=None)
        else:
            """Sort(max_age=1, min_hits=3, iou_threshold=0.3)
            --max_age: Maximum number of frames to keep alive a track without associated detections. type=int, default=1
            --min_hits: Minimum number of associated detections before track is initialised. type=int, default=3
            --iou_threshold: Minimum IOU for match., type=float, default=0.3
            """
            self.id_tracker = Sort(max_age=32, min_hits=1, iou_threshold=0.2)

    def match_detections(self, frame, bboxes, labels, confidences):
        """
        frame: np.array((h,w,3)) image
        bboxes: list of int [x1,y1,x2,y2]
        labels: list of strings
        confidences: list of int [0-100] values
        """
        if DEEP_SORT:
            detections = []
            for bbox, conf, label in zip(bboxes,confidences,labels):
                x0, y0, x1, y1 = [int(r) for r in bbox]
                detections.append( ([x0,y0,x1-x0,y1-y0],sorted([0,conf,100])[1]/100.,label) )

            result = self.id_tracker.update_tracks(detections, frame=frame)

            for track, label, confidence in zip(result,labels,confidences):
                if not track.is_confirmed():
                    continue
                id = track.track_id
                x0,y0,x1,y1 = [int(b) for b in track.to_ltrb()]

                if id in self.__instances:
                    instance = self.__instances[id]
                    instance.set_bbox(frame, (x0,y0,x1,y1))
                    instance.label = str(label)
                    instance.confidence = int(confidence)
                else:
                    instance = TrackedObject( id, str(label), int(confidence) )
                    instance.set_bbox(frame, (x0,y0,x1,y1))
                    with self.instances_lock:
                        self.__instances[id] = instance

        else:
            detections = np.empty( (0,5) )
            for bbox, conf in zip(bboxes,confidences):
                currentArray = np.array( [*bbox, sorted([0,conf,100])[1]/100.] )
                detections = np.vstack( (detections, currentArray) )

            result = self.id_tracker.update(detections)

            for res, label, confidence in zip(result,labels,confidences):
                x0, y0, x1, y1, id = [int(r) for r in res]

                if id in self.__instances:
                    instance = self.__instances[id]
                    instance.set_bbox(frame, (x0,y0,x1,y1))
                    instance.label = str(label)
                    instance.confidence = int(confidence)
                else:
                    instance = TrackedObject( id, str(label), int(confidence) )
                    instance.set_bbox(frame, (x0,y0,x1,y1))
                    with self.instances_lock:
                        self.__instances[id] = instance

            detected = set(int(r[4]) for r in result)
            for id in set(self.__instances.keys())-detected:
                self.__instances[id].status = TrackedObject.STS_LOST
            for id in list(self.__instances.keys()):
                if self.__instances[id].status == TrackedObject.STS_LOST:
                    with self.instances_lock:
                        self.__instances.pop(id)

    def get_detections(self, frame):
        if not self.objects:
            return []

        ids,bboxs,labels,confidences = zip(*([obj.id, obj.get_bbox(frame), obj.label, obj.confidence] for obj in self.objects))

        if not DEEP_SORT:
            detections = np.empty( (0,5) )
            for bbox,confidence in zip(bboxs,confidences):
                currentArray = np.array( [*bbox, confidence] )
                detections = np.vstack( (detections, currentArray) )
            self.id_tracker.update(detections)

        return zip(ids,bboxs,labels)


    @property
    def objects(self):
        with self.instances_lock:
            ans = set(self.__instances.values())
        return ans