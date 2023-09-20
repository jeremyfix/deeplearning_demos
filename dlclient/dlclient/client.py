#!/usr/bin/env python3
# coding: utf-8

# This file is part of dlclient.

# dlclient is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlclient is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlclient. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import socket
from enum import Enum

# External imports
from whiptail import Whiptail

# Local imports
from dlclient import utils

ClientStates = Enum("ClientStates", ["INIT", "SELECT", "QUIT", "FINAL"])


class Client:

    MASTER_COMMAND_LENGTH = 4
    MSG_LENGTH_NUMBYTES = 7

    def __init__(self, hostname, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((hostname, port))
        self.current_state = ClientStates.INIT
        self.keep_running = True
        self.tmp_buf = bytearray(self.MSG_LENGTH_NUMBYTES)
        self.tmp_view = memoryview(self.tmp_buf)

        self.data_buf = bytearray(9999999)
        self.data_view = memoryview(self.data_buf)

        self.selected_model = None

        self.whiptail = Whiptail(
            title="Deep learning client", backtitle="Nothing special :) "
        )

        self.callbacks = {
            ClientStates.INIT: self.get_list,
            ClientStates.QUIT: self.quit,
            ClientStates.SELECT: self.select,
        }

    def get_list(self):
        request = bytes("list", "ascii")
        utils.send_data(self.sock, request)
        cmd = utils.recv_data(self.sock, self.MASTER_COMMAND_LENGTH).decode("ascii")
        if cmd != "list":
            raise RuntimeError("Unexpected server reply")
        # Read the message length
        utils.recv_data_into(self.sock, self.tmp_view, self.MSG_LENGTH_NUMBYTES)
        msg_length = int(self.tmp_buf.decode("ascii"))

        # Read the message
        utils.recv_data_into(self.sock, self.data_view[:msg_length], msg_length)
        model_list = self.data_buf.decode("ascii")[:msg_length].split("\n")

        self.selected_model, cancel = self.whiptail.menu(
            "Select the model you want to run. Cancel to quit", model_list
        )
        if cancel:
            return ClientStates.QUIT
        else:
            return ClientStates.SELECT

    def select(self):
        return ClientStates.INIT

    def quit(self):
        request = bytes("quit", "ascii")
        utils.send_data(self.sock, request)
        self.keep_running = False
        return ClientStates.FINAL

    def run(self):
        while self.keep_running:
            self.current_state = self.callbacks[self.current_state]()
