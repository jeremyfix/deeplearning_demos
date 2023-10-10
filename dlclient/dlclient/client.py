#!/usr/bin/env python3
# coding: utf-8

# This file is part of dlclient.

# dlclient is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlclient is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlclient. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import socket
from enum import Enum
from typing import List

# External imports
from whiptail import Whiptail
import cv2

# Local imports
from dlclient import utils, video_grabber

ClientStates = Enum(
    "ClientStates", ["INIT", "SELECT", "PREPARE", "FRAME", "QUIT", "FINAL"]
)

# Number of bytes for encoding the command and the message length
MSG_LENGTH_NUMBYTES = 6
ENDIANESS = "big"
STR_ENCODING = "ascii"
# Mapping from the human readable ascii commands to their byte code
MASTER_COMMAND_LENGTH = 1  # To be adjusted if need, according to below
COMMANDS_ENCODINGS = {
    "list": 0b0001,
    "quit": 0b0010,
    "select": 0b0011,
    "ready": 0b0100,
    "input": 0b0101,
    "output": 0b0110,
    "data": 0b0111,
    "result": 0b1000,
}


class CommandParser:
    def __init__(self):
        self.tmp_buffer = bytearray(max(MASTER_COMMAND_LENGTH, MSG_LENGTH_NUMBYTES))
        self.tmp_view = memoryview(self.tmp_buffer)

        self.data_buf = bytearray(9999999)
        self.data_view = memoryview(self.data_buf)

        command_coding = list(COMMANDS_ENCODINGS.items())
        for c, v in command_coding:
            COMMANDS_ENCODINGS[v] = c

    def read_command(self, request):
        utils.recv_data_into(
            request,
            self.tmp_view[:MASTER_COMMAND_LENGTH],
            MASTER_COMMAND_LENGTH,
        )
        cmd = self.tmp_buffer[:MASTER_COMMAND_LENGTH]

        return cmd

    def read_data(self, request):
        # Read the num bytes of the data
        utils.recv_data_into(request, self.tmp_view, MSG_LENGTH_NUMBYTES)
        msg_length = int.from_bytes(self.tmp_buffer, ENDIANESS)

        # Read the message
        utils.recv_data_into(request, self.data_view[:msg_length], msg_length)

        return self.data_buf[:msg_length]


def read_command(request, expected: List[str] = None):
    parser = CommandParser()
    cmd_int = int.from_bytes(parser.read_command(request), ENDIANESS)
    cmd = COMMANDS_ENCODINGS[cmd_int]
    if expected is not None and cmd not in expected:
        raise RuntimeError(
            f"Error in the protocol, expected the command in {expected}, but got {cmd}"
        )
    return cmd


def read_data(request):
    parser = CommandParser()
    return parser.read_data(request)


def send_command(request, cmd):
    cmd = COMMANDS_ENCODINGS[cmd].to_bytes(MASTER_COMMAND_LENGTH, ENDIANESS)
    utils.send_data(request, cmd)


def send_data(request, msg):
    msg_len = len(msg)
    msg_len = msg_len.to_bytes(MSG_LENGTH_NUMBYTES, ENDIANESS)
    utils.send_data(request, msg_len)
    utils.send_data(request, msg)


class VideoInputProvider:
    def __init__(self, jpeg_quality, jpeg_lib, resize_factor, device_id):
        self.video_grabber = video_grabber.VideoGrabber(
            jpeg_quality=jpeg_quality,
            jpeg_lib=jpeg_lib,
            resize=resize_factor,
            device_id=device_id,
        )
        self.video_grabber.start()

    def release(self):
        self.video_grabber.stop()
        self.video_grabber.join()

    def get_data(self):
        img_buffer, orig_img = self.video_grabber.get_buffer()
        while img_buffer is None:
            img_buffer, orig_img = self.video_grabber.get_buffer()
        return img_buffer


class VideoOutputSaver:
    def __init__(self, jpeg_lib):
        self.jpeg_handler = utils.make_jpeg_handler(jpeg_lib, 100)

    def display(self, data):
        # data is a byte array
        np_img = self.jpeg_handler.decompress(data)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("img.jpg", np_img)
        return False


class VideoOutputDisplayer:
    def __init__(self, jpeg_lib):
        self.jpeg_handler = utils.make_jpeg_handler(jpeg_lib, 100)
        self.window_name = "Result"
        self.window = cv2.namedWindow("Result")
        # cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
        )

    def __del__(self):
        cv2.destroyWindow(self.window_name)

    def display(self, data):
        np_img = self.jpeg_handler.decompress(data)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, np_img)
        keep_running = not (cv2.waitKey(1) & 0xFF == ord("q"))
        return keep_running


class Client:
    def __init__(self, hostname, port, device_id, resize_factor):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((hostname, port))
        self.current_state = ClientStates.INIT
        self.keep_running = True

        self.selected_model = None

        self.whiptail = Whiptail(
            title="Deep learning client", backtitle="Nothing special :) "
        )

        self.callbacks = {
            ClientStates.INIT: self.get_list,
            ClientStates.PREPARE: self.prepare,
            ClientStates.FRAME: self.frame,
            ClientStates.QUIT: self.quit,
            ClientStates.SELECT: self.select,
        }

        # Parameters if we use a webcam provider
        self.jpeg_lib = "cv2"
        self.jpeg_quality = 100
        self.resize_factor = resize_factor
        self.device_id = device_id

        self.input_type = None
        self.input_provider = None
        self.output_type = None
        self.output_displayer = None

    def get_list(self):
        send_command(self.sock, "list")
        _ = read_command(self.sock, ["list"])
        model_list = read_data(self.sock).decode(STR_ENCODING).split("\n")

        self.selected_model, cancel = self.whiptail.menu(
            "Select the model you want to run. Cancel to quit", model_list
        )
        if cancel:
            return ClientStates.QUIT
        else:
            return ClientStates.SELECT

    def select(self):
        # Send to the server the selected model
        msg = bytes(self.selected_model, STR_ENCODING)

        send_command(self.sock, "select")
        send_data(self.sock, msg)

        # wait until the server is ready to get
        _ = read_command(self.sock, ["ready"])

        return ClientStates.PREPARE

    def prepare(self):
        _ = read_command(self.sock, ["input"])
        self.input_type = read_data(self.sock).decode(STR_ENCODING)
        _ = read_command(self.sock, ["output"])
        self.output_type = read_data(self.sock).decode(STR_ENCODING)

        if self.input_type == "image":
            self.input_provider = VideoInputProvider(
                self.jpeg_quality, self.jpeg_lib, self.resize_factor, self.device_id
            )
        elif self.input_type == "text":
            raise NotImplementedError
        else:
            raise RuntimeError(
                f"I do not know what to do to provide input data of type {self.input_type}"
            )

        if self.output_type == "image":
            self.output_displayer = VideoOutputDisplayer(self.jpeg_lib)
        elif self.input_type == "text":
            raise NotImplementedError
        else:
            raise RuntimeError(
                f"I do not know what to do to provide input data of type {self.input_type}"
            )

        self.keep_on_sending_frame = True
        return ClientStates.FRAME

    def frame(self):
        # Decision: do we send another frame ?
        if self.keep_on_sending_frame:
            # and then send the frame
            send_command(self.sock, "data")
            send_data(self.sock, self.input_provider.get_data())

            # And wait for the result
            _ = read_command(self.sock, ["result"])

            result = read_data(self.sock)
            self.keep_on_sending_frame = self.output_displayer.display(result)

            # And loop back to frame
            return ClientStates.FRAME
        else:
            # Release the input grabber
            self.input_provider.release()
            del self.input_provider
            self.input_provider = None
            del self.output_displayer
            self.output_displayer = None

            # And tell the server to stop processing frames
            send_command(self.sock, "quit")

            # And loop back to the list request of the available models
            return ClientStates.INIT

    def quit(self):
        send_command(self.sock, "quit")
        self.keep_running = False
        return ClientStates.FINAL

    def run(self):
        while self.keep_running:
            self.current_state = self.callbacks[self.current_state]()
