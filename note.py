import cv2

from hu import classify_clef
from util import distance

violin_key = {
    0: 'F5',
    1: 'E5',
    2: 'D5',
    3: 'C5',
    4: 'H4',
    5: 'A4',
    6: 'G4',
    7: 'F4',
    8: 'E4'
}

bass_key = {
    0: 'A3',
    1: 'G3',
    2: 'F3',
    3: 'E3',
    4: 'D3',
    5: 'C3',
    6: 'H2',
    7: 'A2',
    8: 'G2'
}


def extract_notes(blobs, staffs, image):
    notes = []
    for blob in blobs:
        if blob[1] % 2 == 1:
            staff_no = int((blob[1] - 1) / 2)
            notes.append(Note(staff_no, staffs[staff_no], blob[0], image))
    return notes


def draw_notes_pitch(image, notes):
    im_with_pitch = image.copy()
    im_with_pitch = cv2.cvtColor(im_with_pitch, cv2.COLOR_GRAY2BGR)
    for note in notes:
        cv2.putText(im_with_pitch, note.pitch, (int(note.center[0]) - 10, int(note.center[1]) + 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 0))
    cv2.imwrite('output/9_with_pitch.jpg', im_with_pitch)


# noinspection PyMethodMayBeStatic
class Note:
    """
    Represents a single note
    """
    def __init__(self, staff_no, staff, blob, image):
        self.position_on_staff = self.detect_position_on_staff(staff, blob)
        self.staff_no = staff_no
        self.center = blob.pt
        self.key, self.pitch = self.detect_pitch(image, staff, self.position_on_staff)

    def detect_position_on_staff(self, staff, blob):
        distances_from_lines = []
        x, y = blob.pt
        for line_no, line in enumerate(staff.lines_location):
            distances_from_lines.append((2 * line_no, distance((x, y), (x, line))))
        distances_from_lines = sorted(distances_from_lines, key=lambda tup: tup[1])
        # Check whether difference between two closest distances is within MIDDLE_SNAPPING value specified in config.py
        if distances_from_lines[1][1] - distances_from_lines[0][1] <= staff.lines_distance / 2:
            # Place the note between these two lines
            return int((distances_from_lines[0][0] + distances_from_lines[1][0]) / 2)
        else:
            # Place the note on the line closest to blob's center
            return distances_from_lines[0][0]

    def detect_pitch(self, image, staff, position_on_staff):
        if classify_clef(image, staff) == 'violin':
            return 'violin', violin_key[position_on_staff]
        else:
            return 'bass', bass_key[position_on_staff]


