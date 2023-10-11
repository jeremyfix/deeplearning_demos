# coding: utf-8 This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import logging
from enum import Enum
from threading import Lock
import socket
from typing import List

# Local imports
from dlserver import utils, preprocessing, postprocessing, models


MasterStates = Enum("MasterStates", ["INIT", "FINAL"])


class CommandParserSingletonMeta(type):
    _instance = None
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                instance = super().__call__(*args, **kwargs)
                cls._instance = instance
        return cls._instance


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


class CommandParser(metaclass=CommandParserSingletonMeta):
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


def send_data(request: socket.socket, msg):
    msg_len = len(msg)
    msg_len = msg_len.to_bytes(MSG_LENGTH_NUMBYTES, ENDIANESS)
    utils.send_data(request, msg_len)
    utils.send_data(request, msg)


def get_host_id(sock: socket.socket):
    hostname, port = sock.getsockname()
    return f"{hostname}:{port}"


class MasterStateMachine:
    def __init__(self, models):
        self.keeps_running = True
        self.models = models

        self.current_state = MasterStates.INIT
        self.transitions = {
            MasterStates.INIT: {
                "list": self.on_list,
                "quit": self.on_quit,
                "select": self.on_select,
            }
        }

    def on_list(self, request):
        logging.debug("on_list")
        send_command(request, "list")
        send_data(request, bytes("\n".join([m for m in self.models]), STR_ENCODING))
        return MasterStates.INIT

    def on_quit(self, request):
        logging.debug("on_quit")
        self.keeps_running = False
        return MasterStates.FINAL

    def on_select(self, request):
        logging.debug("on_select")

        model_name = read_data(request).decode(STR_ENCODING)
        logging.debug(f"Loading {model_name} for {get_host_id(request)}")

        # We delegate the FSM to the sub-FSM of the model
        model_fsm = ModelStateMachine(request, model_name, self.models[model_name])
        model_fsm.step()

        return MasterStates.INIT

    def step(self, request):
        try:
            while self.current_state != MasterStates.FINAL:
                logging.debug(f"In state {self.current_state}")

                # Read the command
                allowed_commands = self.transitions[self.current_state]
                cmd = read_command(request, allowed_commands.keys())
                self.current_state = allowed_commands[cmd](request)
        except Exception as e:
            logging.error(
                f"Connection was closed. Exception was '{e}'. I'm quitting the thread"
            )
            self.current_state = MasterStates.FINAL
            self.keeps_running = False


ModelStates = Enum(
    "ModelStates",
    ["INIT", "READY", "PREPROCESS", "PROCESS", "POSTPROCESS", "RELEASE", "FINAL"],
)


class ModelStateMachine:
    def __init__(self, request, model_name: str, model: dict):
        """
        request: socket.socket
            the socket to discuss with the client

        model_name: str
        model: dict
            model:
                cls:
                params:
            preprocessing:
            input_type
            postprocessing:
            output_type:

        """
        self.current_state = ModelStates.INIT
        self.request = request
        self.callbacks = {
            ModelStates.INIT: self.on_init,
            ModelStates.READY: self.on_ready,
            ModelStates.PREPROCESS: self.on_preprocess,
            ModelStates.PROCESS: self.on_process,
            ModelStates.POSTPROCESS: self.on_postprocess,
            ModelStates.RELEASE: self.on_release,
        }

        self.input_data = None
        self.preprocessed = None

        model_cls = model["model"]["cls"]
        model_params = model["model"]["params"]
        self.model = models.load_model(model_cls, model_name, model_params)

        self.frame_assets = {}
        self.input_type = model["input_type"]
        self.fn_preprocessing = preprocessing.load_function(model["preprocessing"])
        self.fn_postprocessing = postprocessing.load_function(
            model["postprocessing"]["cls"], model["postprocessing"]["params"]
        )
        self.output_type = model["output_type"]

        # For compressing/decompressing JPEG images
        self.jpeg_handler = utils.make_jpeg_handler("cv2", 100)

    def on_init(self):
        # Prepare the model, i.e. preload all the things
        # we need to do the job
        # Preprocessing, postprocessing functions
        # and the model (e.g. onnx download and load)

        logging.debug("init")
        # If success, loop to READY
        # TODO : if fail, loop to release

        # We are now ready and we indicate the client we are ready
        send_command(self.request, "ready")

        # Send the expected input type
        # as well as the output type
        send_command(self.request, "input")
        send_data(self.request, bytes(self.input_type, STR_ENCODING))
        send_command(self.request, "output")
        send_data(self.request, bytes(self.output_type, STR_ENCODING))

        return ModelStates.READY

    def on_ready(self):
        # Listen for the command either data or quit

        logging.debug("ready")
        # Wait for the next command
        cmd = read_command(self.request, ["quit", "data"])
        if cmd == "quit":
            return ModelStates.RELEASE
        else:
            # cmd == data
            # Get the data
            received_data = read_data(self.request)
            self.frame_assets = {}

            # Decompress the input data if needed
            if self.input_type == "image":
                self.input_data = received_data

                self.input_data = self.jpeg_handler.decompress(received_data)
                logging.debug(
                    f"Got a frame of type {type(self.input_data)}, shape {self.input_data.shape}"
                )
                self.frame_assets["src_img"] = self.input_data.copy()
            elif self.input_type == "text":
                raise NotImplementedError
            else:
                raise RuntimeError(
                    f"I do not know what to process incoming data of type {self.input_type}"
                )

            # And then transit to preprocess
            return ModelStates.PREPROCESS

    def on_preprocess(self):
        # We got some data, we need to preprocess them
        logging.debug("preprocessing")
        self.preprocessed = self.fn_preprocessing(self.input_data, self.frame_assets)
        return ModelStates.PROCESS

    def on_process(self):
        # We got a preprocesse data, we need to perform inference
        # with the neural net
        logging.debug(f"processing input of type {self.preprocessed.dtype}")
        self.model(self.preprocessed, self.frame_assets)
        return ModelStates.POSTPROCESS

    def on_postprocess(self):
        # We got the output of the model,
        # we need to postprocess the result, send it to the client
        # and loop back to the READY state
        logging.debug("postprocessing")
        result = self.fn_postprocessing(self.frame_assets)
        if self.output_type == "image":
            result = self.jpeg_handler.compress(result)[0]
        elif self.output_type == "text":
            raise NotImplementedError
        else:
            raise RuntimeError(
                f"I do not know what to process outgoing data of type {self.output_type}"
            )

        send_command(self.request, "result")
        send_data(self.request, result)
        return ModelStates.READY

    def on_release(self):
        # We are asked to stop using this model
        # We release the data and die
        return ModelStates.FINAL

    def step(self):
        while self.current_state != ModelStates.FINAL:
            self.current_state = self.callbacks[self.current_state]()
