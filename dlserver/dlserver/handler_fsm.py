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

    MASTER_COMMAND_LENGTH = 5

    def __init__(self, models):
        self.tmp_buffer = bytearray(self.MASTER_COMMAND_LENGTH)
        self.tmp_view = memoryview(self.tmp_buffer)

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
        model_list = bytes("\n".join([m["name"] for m in self.models]), "ascii")
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
                cmd = (
                    self.tmp_buffer[: self.MASTER_COMMAND_LENGTH]
                    .decode("ascii")
                    .rstrip()
                )
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
    "ModelStates", ["INIT", "READY", "PREPROCESS", "PROCESS", "POSTPROCES", "FINAL"]
)


class ModelStateMachine:
    def __init__(self, request):
        self.current_state = ModelStates.INIT
        self.request = request

    def on_init(self, request):
        # Prepare the model, i.e. preload all the things
        # we need to do the job
        # Preprocessing, postprocessing functions
        # and the model (e.g. onnx download and load)
        pass

    def on_reasy(self, request):
        # Listen for the command either data or quit
        pass

    def on_preprocess(self, request):
        # We got some data, we need to preprocess them
        pass

    def on_process(self, request):
        # We got a preprocesse data, we need to perform inference
        # with the neural net
        pass

    def on_postprocess(self, request):
        # We got the output of the model,
        # we need to postprocess the result, send it to the client
        # and loop back to the READY state
        pass

    def on_release(self, request):
        # We are asked to stop using this model
        # We release the data and die
        pass

    def step(self):
        pass
