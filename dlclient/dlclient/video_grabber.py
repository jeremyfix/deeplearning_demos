# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard modules
from threading import Thread, Lock

# External modules
import cv2

# Local modules
from dlclient import utils


class VideoGrabber(Thread):
    """A threaded video grabber.

    Attributes:
    encode_params ():
    cap (str):
    attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, jpeg_quality, jpeg_lib, resize, device_id=0):
        """Constructor.

        Args:
        jpeg_quality (:obj:`int`): Quality of JPEG encoding, in 0, 100.
        resize (:obj:`float'): resize factor in [0, 1]

        """
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(device_id)
        self.resize_factor = resize
        self.running = True
        self.buffer = None
        self.lock = Lock()

        self.jpeg_handler = utils.make_jpeg_handler(jpeg_lib, jpeg_quality)

    def stop(self):
        self.running = False

    def get_buffer(self):
        """Method to access the encoded buffer.

        Returns:
        np.ndarray: the compressed image if one has been acquired.
                    None otherwise.
        """
        if self.buffer is not None:
            self.lock.acquire()
            cpy_buffer = self.buffer
            self.lock.release()
            return cpy_buffer  # holds a pair of the compressed image and original image
        else:
            return None, None

    def run(self):
        while self.running:
            success, img = self.cap.read()
            target_size = (
                int(img.shape[1] * self.resize_factor),
                int(img.shape[0] * self.resize_factor),
            )
            img = cv2.resize(img, target_size)
            if not success:
                continue

            # JPEG compression
            # Protected by a lock
            # As the main thread may asks to access the buffer
            self.lock.acquire()
            self.img = img
            self.buffer = self.jpeg_handler.compress(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            )
            self.lock.release()
