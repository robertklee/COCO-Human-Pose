"""Automatic person detection used to crop uploaded images before pose estimation.

The pose model is trained on crops that are centred on a single person occupying most
of the frame. This module locates people with a small COCO-trained object detector run
through ``cv2.dnn`` (no extra Python dependency), so the demo app can crop
automatically instead of relying on the user to pre-crop their photo.

Every entry point degrades gracefully: if the weights cannot be downloaded or the
network cannot be built, detection returns no results and the caller falls back to the
previous whole-image behaviour.
"""

import logging
import os
import tempfile
import threading

import cv2
import numpy as np
import requests

from constants import *

logger = logging.getLogger(__name__)

_detector_lock = threading.Lock()
_detector = None
_detector_failed = False


def _weights_path():
    return os.path.join(DEFAULT_MODEL_BASE_DIR, PERSON_DETECTOR_FILENAME)


def _download_weights(destination):
    """Download the detector weights, trying each mirror in turn.

    The download is staged in a temporary file and moved into place only once it is
    complete, so an interrupted download can never leave a corrupt cache behind.

    Returns True if the weights are present at *destination* afterwards.
    """
    if os.path.exists(destination) and os.path.getsize(destination) >= PERSON_DETECTOR_MIN_BYTES:
        return True

    os.makedirs(os.path.dirname(destination) or '.', exist_ok=True)

    for url in PERSON_DETECTOR_URLS:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(destination) or '.',
                                             suffix='.part') as tmp:
                tmp_path = tmp.name
                with requests.get(url, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        tmp.write(chunk)

            if os.path.getsize(tmp_path) < PERSON_DETECTOR_MIN_BYTES:
                raise ValueError(f'downloaded file is too small to be valid weights '
                                 f'({os.path.getsize(tmp_path)} bytes)')

            os.replace(tmp_path, destination)
            tmp_path = None
            return True
        except Exception:
            logger.warning('Could not download person detector weights from %s', url, exc_info=True)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return False


def _generate_anchors():
    """Pre-compute the YOLOX grid centres and per-cell strides for each feature level."""
    grids = []
    expanded_strides = []

    for stride in PERSON_DETECTOR_STRIDES:
        hsize = PERSON_DETECTOR_INPUT_DIM[1] // stride
        wsize = PERSON_DETECTOR_INPUT_DIM[0] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))

    return np.concatenate(grids, 1), np.concatenate(expanded_strides, 1)


class PersonDetector():
    """Thin ``cv2.dnn`` wrapper around a YOLOX detector, filtered to the person class."""

    def __init__(self, weights_path) -> None:
        self.net = cv2.dnn.readNet(weights_path)
        self.grids, self.expanded_strides = _generate_anchors()

    def _letterbox(self, image):
        """Resize *image* into the detector's square input, preserving aspect ratio.

        Returns the padded image and the scale factor applied, so detections can be
        mapped back to the source image coordinates.
        """
        target_w, target_h = PERSON_DETECTOR_INPUT_DIM
        h, w = image.shape[:2]
        scale = min(target_h / h, target_w / w)
        resized_h, resized_w = int(h * scale), int(w * scale)

        padded = np.full((target_h, target_w, 3), PERSON_DETECTOR_PAD_VALUE, dtype=np.float32)
        padded[:resized_h, :resized_w] = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        return padded, scale

    def detect(self, image):
        """Detect people in an RGB uint8 image.

        Returns a list of ``(x, y, w, h, confidence)`` tuples in *image* coordinates,
        sorted by descending confidence. Returns an empty list on any failure.
        """
        try:
            padded, scale = self._letterbox(image)

            self.net.setInput(np.transpose(padded, (2, 0, 1))[np.newaxis, ...])
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

            return self._postprocess(outputs[0][0], scale)
        except Exception:
            logger.warning('Person detection failed; falling back to the whole image',
                           exc_info=True)
            return []

    def _postprocess(self, predictions, scale):
        """Decode raw YOLOX predictions into person boxes in source-image coordinates."""
        # Decode box centres/sizes from grid-relative offsets into detector-input pixels
        centers = (predictions[:, :2] + self.grids[0]) * self.expanded_strides[0]
        sizes = np.exp(predictions[:, 2:4]) * self.expanded_strides[0]

        boxes_xywh = np.concatenate([centers - sizes / 2, sizes], axis=1)

        # Objectness times the person class probability
        scores = predictions[:, 4] * predictions[:, 5 + PERSON_DETECTOR_PERSON_CLASS_ID]
        # Only keep boxes where 'person' is the winning class, so that e.g. a horse
        # scoring slightly on the person class does not produce a spurious detection.
        best_class = np.argmax(predictions[:, 5:5 + PERSON_DETECTOR_NUM_CLASSES], axis=1)

        candidates = np.where((scores > PERSON_DETECTOR_CONF_THRESHOLD) &
                              (best_class == PERSON_DETECTOR_PERSON_CLASS_ID))[0]
        if len(candidates) == 0:
            return []

        boxes_xywh = boxes_xywh[candidates]
        scores = scores[candidates]

        keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(),
                                PERSON_DETECTOR_CONF_THRESHOLD, PERSON_DETECTOR_NMS_THRESHOLD)
        if len(keep) == 0:
            return []

        keep = np.array(keep).flatten()
        # Undo the letterbox scaling to get back to source-image coordinates
        boxes = boxes_xywh[keep] / scale
        confidences = scores[keep]

        detections = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                      for b, c in zip(boxes, confidences)]
        detections.sort(key=lambda d: d[4], reverse=True)

        return detections


def get_detector():
    """Return the cached detector, or None if it is unavailable.

    The weights are downloaded and the network is built at most once per process; a
    failure is remembered so repeated calls do not retry the download on every image.
    """
    global _detector, _detector_failed

    with _detector_lock:
        if _detector is not None or _detector_failed:
            return _detector

        try:
            weights_path = _weights_path()
            if not _download_weights(weights_path):
                raise RuntimeError('person detector weights are unavailable')

            _detector = PersonDetector(weights_path)
        except Exception:
            logger.warning('Person detector unavailable; automatic cropping is disabled',
                           exc_info=True)
            _detector_failed = True

        return _detector


def detect_people(image):
    """Detect people in an RGB uint8 image.

    Returns a list of ``(x, y, w, h, confidence)`` tuples, sorted by descending
    confidence, or an empty list if detection is unavailable or finds nobody.
    """
    detector = get_detector()
    if detector is None:
        return []

    return detector.detect(image)


def select_primary_person(detections, image_width, image_height):
    """Pick the single detection the pose model should run on.

    Detections are scored by a combination of area, closeness to the image centre, and
    confidence. The pose model is trained to label the person in the centre of the crop,
    so a large off-centre bystander should not beat the obvious subject of the photo.

    Returns the winning ``(x, y, w, h, confidence)`` tuple, or None if *detections* is empty.
    """
    if not detections:
        return None

    image_area = max(1.0, float(image_width) * float(image_height))
    # Largest possible centre offset, used to normalize the distance into [0, 1]
    max_offset = max(1e-6, np.hypot(image_width / 2.0, image_height / 2.0))

    def score(detection):
        x, y, w, h, confidence = detection
        area_term = (max(0.0, w) * max(0.0, h) / image_area) ** PERSON_AREA_EXPONENT
        offset = np.hypot((x + w / 2.0) - image_width / 2.0,
                          (y + h / 2.0) - image_height / 2.0)
        center_term = max(0.0, 1.0 - offset / max_offset) ** PERSON_CENTER_EXPONENT
        return area_term * center_term * confidence

    # max() keeps the first of any tied detections, and detections are confidence-sorted,
    # so selection is deterministic.
    return max(detections, key=score)
