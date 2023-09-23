# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import logging
from enum import Enum
from threading import Lock
import socket

# Local imports
from dlserver import utils


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
    "list": 0b001,
    "quit": 0b010,
    "select": 0b011,
    "ready": 0b100,
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


def read_command(request):
    parser = CommandParser()
    cmd_int = int.from_bytes(parser.read_command(request), ENDIANESS)
    cmd = COMMANDS_ENCODINGS[cmd_int]
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
        model_fsm = ModelStateMachine(request, self.models[model_name])
        model_fsm.step()

        return MasterStates.INIT

    def step(self, request):
        try:
            while self.current_state != MasterStates.FINAL:
                logging.debug(f"In state {self.current_state}")

                # Read the command
                cmd = read_command(request)
                logging.debug(f"got {cmd}")
                allowed_commands = self.transitions[self.current_state]
                if cmd in allowed_commands:
                    self.current_state = allowed_commands[cmd](request)
                else:
                    logging.error(f"Got an unrecognized command {cmd}")
        except RuntimeError as e:
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
    def __init__(self, request, model: dict):
        """
        request: socket.socket
            the socket to discuss with the client

        model: dict
            name:
            url:
            input_type
            preprocessing:
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

    def on_init(self):
        # Prepare the model, i.e. preload all the things
        # we need to do the job
        # Preprocessing, postprocessing functions
        # and the model (e.g. onnx download and load)

        # If success, loop to READY
        # TODO : if fail, loop to release
        return ModelStates.READY

    def on_ready(self):
        # Listen for the command either data or quit
        # We are now ready and we indicate the client we are ready
        send_command(self.request, "ready")
        return ModelStates.PREPROCESS

    def on_preprocess(self):
        # We got some data, we need to preprocess them
        return ModelStates.PROCESS

    def on_process(self):
        # We got a preprocesse data, we need to perform inference
        # with the neural net
        return ModelStates.POSTPROCESS

    def on_postprocess(self):
        # We got the output of the model,
        # we need to postprocess the result, send it to the client
        # and loop back to the READY state
        return ModelStates.FINAL

    def on_release(self):
        # We are asked to stop using this model
        # We release the data and die
        return ModelStates.FINAL

    def step(self):
        while self.current_state != ModelStates.FINAL:
            self.current_state = self.callbacks[self.current_state]()
