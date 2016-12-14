import cv2

from config import NOTE_PITCH_DETECTION_MIDDLE_SNAPPING, VERBOSE
from hu import classify_clef
from util import distance

violin_key = {
    -6: 'C6',
    -5: 'D6',
    -4: 'C6',
    -3: 'H5',
    -2: 'A5',
    -1: 'G5',
    0:  'F5',
    1:  'E5',
    2:  'D5',
    3:  'C5',
    4:  'H4',
    5:  'A4',
    6:  'G4',
    7:  'F4',
    8:  'E4',
    9:  'D4',
    10: 'C4',
    11: 'H3',
    12: 'A3',
    13: 'G3',
    14: 'F3',
}

bass_key = {
    -6: 'G3',
    -5: 'F3',
    -4: 'E3',
    -3: 'D3',
    -2: 'C3',
    -1: 'H3',
    0:  'A3',
    1:  'G3',
    2:  'F3',
    3:  'E3',
    4:  'D3',
    5:  'C3',
    6:  'H2',
    7:  'A2',
    8:  'G2',
    9:  'F2',
    10: 'E2',
    11: 'D2',
    12: 'C2',
    13: 'H1',
    14: 'A1',
}


def extract_notes(blobs, staffs, image):
    clef = classify_clef(image, staffs[0])
    notes = []
    if VERBOSE:
        print('Detected clef: ' + clef)
        print('Extracting notes from blobs.')
    for blob in blobs:
        if blob[1] % 2 == 1:
            staff_no = int((blob[1] - 1) / 2)
            notes.append(Note(staff_no, staffs, blob[0], clef))
    if VERBOSE:
        print('Extracted ' + str(len(notes)) + ' notes.')
    return notes


def draw_notes_pitch(image, notes):
    im_with_pitch = image.copy()
    im_with_pitch = cv2.cvtColor(im_with_pitch, cv2.COLOR_GRAY2BGR)
    for note in notes:
        cv2.putText(im_with_pitch, note.pitch, (int(note.center[0]) - 5, int(note.center[1]) + 35),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5, color=(255, 0, 0))
    cv2.imwrite('output/9_with_pitch.png', im_with_pitch)


# noinspection PyMethodMayBeStatic
class Note:
    """
    Represents a single note
    """
    def __init__(self, staff_no, staffs, blob, clef):
        self.position_on_staff = self.detect_position_on_staff(staffs[staff_no], blob)
        self.staff_no = staff_no
        self.center = blob.pt
        self.clef = clef
        self.pitch = self.detect_pitch(self.position_on_staff)

    def detect_position_on_staff(self, staff, blob):
        distances_from_lines = []
        x, y = blob.pt
        for line_no, line in enumerate(staff.lines_location):
            distances_from_lines.append((2 * line_no, distance((x, y), (x, line))))
        # Generate three upper lines
        for line_no in range(5, 8):
            distances_from_lines.append((2 * line_no, distance((x, y), (x, staff.min_range + line_no * staff.lines_distance))))
        # Generate three lower lines
        for line_no in range(-3, 0):
            distances_from_lines.append((2 * line_no, distance((x, y), (x, staff.min_range + line_no * staff.lines_distance))))

        distances_from_lines = sorted(distances_from_lines, key=lambda tup: tup[1])
        # Check whether difference between two closest distances is within MIDDLE_SNAPPING value specified in config.py
        if distances_from_lines[1][1] - distances_from_lines[0][1] <= NOTE_PITCH_DETECTION_MIDDLE_SNAPPING:
            # Place the note between these two lines
            return int((distances_from_lines[0][0] + distances_from_lines[1][0]) / 2)
        else:
            # Place the note on the line closest to blob's center
            return distances_from_lines[0][0]

    def detect_pitch(self, position_on_staff):
        if self.clef == 'violin':
            return violin_key[position_on_staff]
        else:
            return bass_key[position_on_staff]
