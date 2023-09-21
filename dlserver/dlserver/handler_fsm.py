# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import logging
from enum import Enum

# Local imports
from dlserver import utils


MasterStates = Enum("MasterStates", ["INIT", "FINAL"])


class MasterStateMachine:

    MASTER_COMMAND_LENGTH = 4
    MSG_LENGTH_NUMBYTES = 7  # be carefull, there are magic numbers below for this value

    def __init__(self, models):
        self.tmp_buffer = bytearray(self.MSG_LENGTH_NUMBYTES)
        self.tmp_view = memoryview(self.tmp_buffer)

        self.data_buf = bytearray(9999999)
        self.data_view = memoryview(self.data_buf)

        self.keeps_running = True
        self.models = models

        self.current_state = MasterStates.INIT
        self.transitions = {
            MasterStates.INIT: {
                "list": self.on_list,
                "quit": self.on_quit,
                "slct": self.on_select,
            }
        }

    def on_list(self, request):
        logging.debug("on_list")
        model_list = bytes("\n".join([m for m in self.models]), "ascii")
        reply = bytes(f"list{len(model_list):07}", "ascii")
        utils.send_data(request, reply)
        utils.send_data(request, model_list)
        return MasterStates.INIT

    def on_quit(self, request):
        logging.debug("on_quit")
        self.keeps_running = False
        return MasterStates.FINAL

    def on_select(self, request):
        logging.debug("on_select")

        # Read the message length
        utils.recv_data_into(request, self.tmp_view, self.MSG_LENGTH_NUMBYTES)
        msg_length = int(self.tmp_buffer.decode("ascii"))

        # Read the message
        utils.recv_data_into(request, self.data_view[:msg_length], msg_length)
        model_name = self.data_buf.decode("ascii")[:msg_length]
        logging.info(f"Loading {model_name}")

        # We delegate the FSM to the sub-FSM of the model
        model_fsm = ModelStateMachine(request, self.models[model_name])
        model_fsm.step()

        return MasterStates.INIT

    def step(self, request):
        try:
            while self.current_state != MasterStates.FINAL:
                logging.info(f"In state {self.current_state}")
                # We listen for the master command
                # NOTE: we do read 5 bytes (not 4 which is the command length)
                # because we test the code with nc
                # which sends the command followed by \n
                # If we test with telnet, it sends \r\n , so we should
                # read 2 extra byets
                utils.recv_data_into(
                    request,
                    self.tmp_view[: self.MASTER_COMMAND_LENGTH],
                    self.MASTER_COMMAND_LENGTH,
                )
                # rstrip is usefull when using nc
                cmd = self.tmp_buffer[: self.MASTER_COMMAND_LENGTH].decode("ascii")
                allowed_commands = self.transitions[self.current_state]
                if cmd in allowed_commands:
                    self.current_state = allowed_commands[cmd](request)
                else:
                    logging.info(f"Got an unrecognized command {cmd}")
        except RuntimeError as e:
            logging.info(
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
        return ModelStates.READY

    def on_ready(self):
        # Listen for the command either data or quit
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
