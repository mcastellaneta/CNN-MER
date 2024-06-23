import time
from IPython.display import display


class Progressbar:
    def __init__(self, id, elements, current_progress=0, bar_length=20,
                 prefix_char='-', progress_char='#', estimated_time=False):
        self.id = id
        self.elements = elements
        self.current_progress = current_progress
        self.bar_length = bar_length
        self.prefix_char = prefix_char
        self.progress_char = progress_char
        self.handle = display(self.print_bar(0), display_id=id)
        self.time_start = None
        self.estimated_time = estimated_time

    def print_bar(self, block):

        bar = (
            f'Progress: [{self.progress_char * block + self.prefix_char * (self.bar_length - block)}]'
        )
        return bar

    def set_elements(self, elements):
        self.elements = elements
        self.update_progress(elements)

    def reset(self):
        self.current_progress = 0
        self.handle.update(self.print_bar(0))

    def add_to_progress(self, progress):

        self.current_progress += progress
        self.update_progress(self.current_progress)

    def update_progress(self, progress):

        if (not self.time_start):
            self.time_start = time.time()

        if not isinstance(progress, int):
            progress = 0
        if progress < 0:
            progress = 0
        if progress > self.elements:
            progress = 1

        self.current_progress = progress
        time_str = ''

        if (self.estimated_time and progress > 0):

            time_remaining = (
                (round(time.time() - self.time_start) / progress)
                * (self.elements - progress)
            )

            mins, sec = divmod(time_remaining, 60)
            time_str = f" Est wait: {int(mins):02}:{sec:05.2f}"


        block = int(round((self.bar_length * progress) / self.elements))
        perc = block / self.bar_length
        text = (
            f'{self.print_bar(block)} {perc * 100:.0f}% \
            {progress}/{self.elements}{time_str}'
        )
        self.handle.update(text)

    def complete(self):

        self.update_progress(self.elements)