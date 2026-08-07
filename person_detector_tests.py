"""Unit tests for automatic person detection, cropping, and coordinate back-mapping.

Run directly with ``python person_detector_tests.py``. These tests exercise only the
geometry, the primary-subject heuristic, and the fail-soft behaviour, so they do not
require the detector weights, the pose model, or the COCO dataset.
"""

import numpy as np

import app_helper
import person_detector
from constants import *
from data_generator import person_bbox_to_crop, transform_bbox_square


def _assert_close(actual, expected, tolerance=1, message=''):
    assert abs(actual - expected) <= tolerance, \
        f'{message}expected {expected}, got {actual}'


def test_crop_is_square_and_centered():
    """A tall box is squared on its longer edge and stays centered on the person."""
    x, y, w, h = 100, 50, 60, 200
    left, upper, right, lower = person_bbox_to_crop((x, y, w, h), vertical_fill=1.0)

    _assert_close(right - left, lower - upper, message='crop is not square: ')
    _assert_close(right - left, h, message='crop edge should match the longer side: ')
    _assert_close((left + right) / 2, x + w / 2, message='crop is not horizontally centered: ')
    _assert_close((upper + lower) / 2, y + h / 2, message='crop is not vertically centered: ')


def test_vertical_fill_controls_person_size():
    """The person occupies the requested fraction of the crop's edge length."""
    h = 200
    for vertical_fill in (0.6, 0.75, 0.9):
        left, upper, right, lower = person_bbox_to_crop((100, 50, 60, h),
                                                        vertical_fill=vertical_fill)
        _assert_close(lower - upper, h / vertical_fill, tolerance=2,
                      message=f'vertical_fill={vertical_fill}: ')
        # The whole person must remain inside the crop
        assert upper <= 50 and lower >= 250, \
            f'vertical_fill={vertical_fill} cropped off part of the person'


def test_wide_pose_is_squared_on_the_wider_edge():
    """A person lying down is squared on width, so no limbs are cropped off."""
    x, y, w, h = 10, 100, 300, 80
    left, upper, right, lower = person_bbox_to_crop((x, y, w, h), vertical_fill=0.75)

    _assert_close(right - left, lower - upper, message='crop is not square: ')
    assert left <= x and right >= x + w, 'crop cut off the width of a wide pose'
    assert upper <= y and lower >= y + h, 'crop cut off the height of a wide pose'


def test_crop_is_not_clamped_to_image_bounds():
    """A person at the edge of the frame produces an out-of-bounds crop, by design.

    Clamping would de-center the subject; PIL zero-pads instead, and the model is
    robust to black borders.
    """
    left, upper, right, lower = person_bbox_to_crop((0, 0, 40, 200), vertical_fill=0.75)

    assert left < 0 and upper < 0, \
        f'crop should extend past the top-left corner, got {(left, upper)}'
    _assert_close(right - left, lower - upper, message='crop is not square: ')


def test_min_edge_floor_expands_about_the_center():
    """A tiny detection is expanded to the floor without moving the person off-center."""
    x, y, w, h = 500, 400, 20, 40
    left, upper, right, lower = person_bbox_to_crop((x, y, w, h), min_edge=INPUT_DIM[0])

    _assert_close(right - left, INPUT_DIM[0], message='min_edge floor not applied: ')
    _assert_close(lower - upper, INPUT_DIM[0], message='min_edge floor not applied: ')
    _assert_close((left + right) / 2, x + w / 2, message='crop is not horizontally centered: ')
    _assert_close((upper + lower) / 2, y + h / 2, message='crop is not vertically centered: ')

    # A box already larger than the floor is left alone
    large = person_bbox_to_crop((0, 0, 800, 800), vertical_fill=1.0, min_edge=INPUT_DIM[0])
    _assert_close(large[2] - large[0], 800, message='min_edge should not shrink a large crop: ')


def test_invalid_vertical_fill_is_rejected():
    for bad in (0, -0.5, 1.5):
        try:
            person_bbox_to_crop((0, 0, 10, 10), vertical_fill=bad)
        except ValueError:
            continue
        raise AssertionError(f'vertical_fill={bad} should have been rejected')


def test_transform_bbox_square_is_unchanged():
    """The existing squaring helper still behaves as the training pipeline expects."""
    assert transform_bbox_square((0, 0, 100, 50)) == (0, -25, 100, 75)


def test_primary_person_prefers_the_centered_subject():
    """A slightly smaller but centered person beats a large one at the edge."""
    width, height = 1000, 1000
    edge_person = (0, 0, 300, 700, 0.9)
    center_person = (400, 200, 250, 600, 0.9)

    chosen = person_detector.select_primary_person(
        [edge_person, center_person], width, height)

    assert chosen == center_person, f'expected the centered person, got {chosen}'


def test_primary_person_prefers_the_larger_of_two_centered_subjects():
    width, height = 1000, 1000
    small = (470, 470, 60, 60, 0.9)
    large = (350, 250, 300, 500, 0.9)

    chosen = person_detector.select_primary_person([small, large], width, height)

    assert chosen == large, f'expected the larger person, got {chosen}'


def test_primary_person_selection_is_deterministic():
    """Equally scoring detections resolve the same way on every call."""
    detections = [(400, 200, 200, 600, 0.9), (400, 200, 200, 600, 0.9)]

    first = person_detector.select_primary_person(detections, 1000, 1000)
    for _ in range(5):
        assert person_detector.select_primary_person(detections, 1000, 1000) == first


def test_no_detections_returns_none():
    assert person_detector.select_primary_person([], 100, 100) is None


def test_detection_failure_is_silent():
    """An unavailable detector yields no detections rather than raising."""
    original = person_detector.get_detector
    person_detector.get_detector = lambda: None
    try:
        assert person_detector.detect_people(np.zeros((10, 10, 3), dtype=np.uint8)) == []
    finally:
        person_detector.get_detector = original


def _round_trip_keypoints(keypoints_256, bbox):
    """Map model-space (256x256) keypoints into original image coordinates.

    Mirrors the mapping performed by ``AppHelper.predict_in_memory_fullres``.
    """
    left, upper, right, lower = bbox
    scale_x = (right - left) / INPUT_DIM[0]
    scale_y = (lower - upper) / INPUT_DIM[1]

    mapped = np.array(keypoints_256, dtype=np.float64)
    mapped[:, 0] = mapped[:, 0] * scale_x + left
    mapped[:, 1] = mapped[:, 1] * scale_y + upper
    return mapped


def test_keypoint_back_mapping_round_trip():
    """A point at a known position in the crop maps back to the same point of the image."""
    bbox = person_bbox_to_crop((300, 100, 200, 600), vertical_fill=0.75)
    left, upper, right, lower = bbox

    # Sample a grid of positions inside the crop, in original image coordinates
    expected = np.array([
        [left + 0.25 * (right - left), upper + 0.25 * (lower - upper)],
        [left + 0.50 * (right - left), upper + 0.50 * (lower - upper)],
        [left + 0.75 * (right - left), upper + 0.90 * (lower - upper)],
    ])
    # The same positions expressed in the model's 256x256 input space
    keypoints_256 = np.array([
        [0.25 * INPUT_DIM[0], 0.25 * INPUT_DIM[1]],
        [0.50 * INPUT_DIM[0], 0.50 * INPUT_DIM[1]],
        [0.75 * INPUT_DIM[0], 0.90 * INPUT_DIM[1]],
    ])

    mapped = _round_trip_keypoints(keypoints_256, bbox)

    assert np.allclose(mapped, expected, atol=1e-6), \
        f'back-mapping mismatch:\n{mapped}\nvs\n{expected}'


def test_keypoint_back_mapping_with_out_of_bounds_crop():
    """Back-mapping stays correct when the crop extends past the top-left corner."""
    bbox = person_bbox_to_crop((0, 0, 100, 400), vertical_fill=0.75)
    assert bbox[0] < 0 and bbox[1] < 0, 'test needs an out-of-bounds crop'

    center = _round_trip_keypoints(
        np.array([[INPUT_DIM[0] / 2, INPUT_DIM[1] / 2]]), bbox)[0]

    _assert_close(center[0], (bbox[0] + bbox[2]) / 2, tolerance=1e-6,
                  message='crop centre maps to the wrong x: ')
    _assert_close(center[1], (bbox[1] + bbox[3]) / 2, tolerance=1e-6,
                  message='crop centre maps to the wrong y: ')


def test_downscale_for_display_scales_crop_and_keypoints_together():
    """Display downscaling keeps keypoints, crop box, and crop size consistent."""
    orig_batch = np.zeros((1, 1000, 800, 3), dtype=np.uint8)
    keypoints = np.array([[[400.0, 500.0, 1.0]]])
    crop_info = {
        'bbox': np.array([-100, -50, 700, 850]),
        'crop_w': 800,
        'crop_h': 900,
        'orig_w': 800,
        'orig_h': 1000,
        'person_bbox': np.array([100, 200, 500, 700]),
    }

    max_dim = 500
    scale = max_dim / 1000
    _resized, kp, ci = app_helper.downscale_for_display(
        orig_batch, keypoints, crop_info, max_dim)

    _assert_close(kp[0, 0, 0], 400 * scale, tolerance=1e-6, message='keypoint x: ')
    _assert_close(kp[0, 0, 1], 500 * scale, tolerance=1e-6, message='keypoint y: ')
    # A negative origin must round, not truncate toward zero, so that
    # bbox[0] + crop_w still lines up with the scaled far edge.
    _assert_close(ci['bbox'][0], -50, message='scaled crop origin x: ')
    _assert_close(ci['bbox'][1], -25, message='scaled crop origin y: ')
    _assert_close(ci['bbox'][0] + ci['crop_w'], ci['bbox'][2],
                  message='scaled crop width is inconsistent with the box: ')
    _assert_close(ci['bbox'][1] + ci['crop_h'], ci['bbox'][3],
                  message='scaled crop height is inconsistent with the box: ')
    _assert_close(ci['person_bbox'][0], 50, message='scaled person box: ')

    # The original crop_info must not be mutated
    assert crop_info['crop_w'] == 800 and crop_info['bbox'][0] == -100


def test_downscale_for_display_is_a_noop_when_already_small():
    orig_batch = np.zeros((1, 100, 100, 3), dtype=np.uint8)
    keypoints = np.array([[[10.0, 20.0, 1.0]]])
    crop_info = {'bbox': np.array([0, 0, 100, 100]), 'crop_w': 100, 'crop_h': 100,
                 'orig_w': 100, 'orig_h': 100}

    batch, kp, ci = app_helper.downscale_for_display(
        orig_batch, keypoints, crop_info, 500)

    assert batch is orig_batch and kp is keypoints and ci is crop_info


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items()) if name.startswith('test_')]

    failures = 0
    for test in tests:
        try:
            test()
            print(f'PASS  {test.__name__}')
        except AssertionError as e:
            failures += 1
            print(f'FAIL  {test.__name__}: {e}')

    print(f'\n{len(tests) - failures}/{len(tests)} tests passed')
    if failures:
        raise SystemExit(1)
