class Note:
    """
    Represents a single note
    """
    def __init__(self, note_image, lines):
        self.idx = None
        self.pitch = None
        self.type = None
        #  TODO setters to check types of variables
        self.image = note_image
        self.lines = lines
        self.base = None  # rectangle

    def detect_pitch(self):
        """
        Detects a pitch of the note
        """
        attached_lines = []
        for line in self.lines:
            if self.base.y_begin <= line.y_begin and self.base.y_end >= line.y_end:
                attached_lines.append(line)

        if len(attached_lines) == 1:
            # Note is exactly on the attached line
            pass
        elif len(attached_lines) == 2 and abs(attached_lines[0].idx - attached_lines[1].idx) == 1:
            # Note is between the two attached lines
            pass
