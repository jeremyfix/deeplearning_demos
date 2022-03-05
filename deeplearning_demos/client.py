#!/usr/bin/env python3

# Standard modules
import argparse
import socket
import sys
import time
import functools

# External modules
import cv2

# Local modules
from deeplearning_demos import video_grabber
from deeplearning_demos import utils


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host", type=str, help="The IP of the echo server", required=True
    )
    parser.add_argument(
        "--port",
        type=int,
        help="The port on which the server is listening",
        required=True,
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        help="The JPEG quality for compressing the reply",
        default=50,
    )
    parser.add_argument(
        "--resize", type=float, help="Resize factor of the image", default=1.0
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["cv2", "turbo"],
        help="Library to use to encode/decode in JPEG the images",
        default="cv2",
    )
    parser.add_argument(
        "--image", type=str, help="Image file to be processed", default=None
    )
    parser.add_argument(
        "--video", type=str, help="Video file to be processed", default=None
    )
    parser.add_argument(
        "--output", type=str, help="Output video filename", default=None
    )
    parser.add_argument(
        "--device", type=int, help="The id of the camera device 0, 1, ..", default=0
    )

    args = parser.parse_args()

    host = args.host
    port = args.port
    jpeg_quality = args.jpeg_quality
    resize_factor = args.resize
    device_id = args.device
    outvideo = None

    cv2.namedWindow("Image")
    cv2.namedWindow("Output")

    keep_running = True

    jpeg_handler = utils.make_jpeg_handler(args.encoder, jpeg_quality)

    if args.image is not None:
        grabber = None
        img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        get_buffer = functools.partial(jpeg_handler.compress, cv2_img=img)
    elif args.video is not None:
        grabber = None
        cap = cv2.VideoCapture(args.video)

        def get_buffer():
            success, img = cap.read()
            return jpeg_handler.compress(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    else:
        grabber = video_grabber.VideoGrabber(
            jpeg_quality, args.encoder, resize_factor, device_id
        )
        grabber.start()
        get_buffer = grabber.get_buffer

    # A temporary buffer in which the received data will be copied
    # this prevents creating a new buffer all the time
    tmp_buf = bytearray(7)
    # this allows to get a reference to a slice of tmp_buf
    tmp_view = memoryview(tmp_buf)

    # Creates a temporary buffer which can hold the largest image we can transmit
    img_buf = bytearray(9999999)
    img_view = memoryview(img_buf)

    idx = 0
    t0 = time.time()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        while keep_running:

            # Grab and encode the image
            img_buffer, orig_img = get_buffer()
            # print(type(orig_img), orig_img.shape)
            if img_buffer is None:
                continue

            # Prepare the message with the number of bytes going to be sent
            msg = bytes("image{:07}".format(len(img_buffer)), "ascii")

            utils.send_data(sock, msg)

            # Send the buffer
            utils.send_data(sock, img_buffer)

            # Read the reply command
            utils.recv_data_into(sock, tmp_view[:5], 5)
            cmd = tmp_buf[:5].decode("ascii")

            if cmd != "image":
                raise RuntimeError("Unexpected server reply")

            # Read the image buffer size
            utils.recv_data_into(sock, tmp_view, 7)
            img_size = int(tmp_buf.decode("ascii"))

            # Read the image buffer
            utils.recv_data_into(sock, img_view[:img_size], img_size)

            # Read the final handshake
            cmd = utils.recv_data(sock, 5).decode("ascii")
            if cmd != "enod!":
                raise RuntimeError(
                    "Unexpected server reply. Expected 'enod!'" ", got '{}'".format(cmd)
                )

            # Transaction is done, we now process/display the received image
            output = jpeg_handler.decompress(img_view[:img_size])
            if len(output.shape) == 3 and output.shape[-1] == 3:
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite("output.jpg", output)
            cv2.imshow("Image", orig_img)
            cv2.imshow("Output", output)

            if args.output is not None:
                if outvideo is None:
                    outvideo = cv2.VideoWriter(
                        args.output,
                        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                        10,
                        (output.shape[1], output.shape[0]),
                    )
                outvideo.write(output)

            keep_running = not (cv2.waitKey(1) & 0xFF == ord("q"))
            if not keep_running:
                sock.sendall("quit!".encode("ascii"))

            idx += 1
            if idx == 30:
                t1 = time.time()
                sys.stdout.write(
                    f"\r {30/(t1-t0):.3} images/second ; msg size : {img_size} ; Image size : {orig_img.shape[0]}x{orig_img.shape[1]}              "
                )
                sys.stdout.flush()
                t0 = t1
                idx = 0
        print()
        print("Closing the socket")
        if grabber is not None:
            print("Stopping the grabber")
            grabber.stop()


if __name__ == "__main__":
    main()
