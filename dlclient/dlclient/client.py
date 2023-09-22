#!/usr/bin/env python3
# coding: utf-8

# This file is part of dlclient.

# dlclient is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlclient is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlclient. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import socket
from enum import Enum
import logging

# External imports
from whiptail import Whiptail

# Local imports
from dlclient import utils

ClientStates = Enum("ClientStates", ["INIT", "SELECT", "QUIT", "FINAL"])

# Number of bytes for encoding the command and the message length
# be carefull, there are magic numbers below for these values
# in send_command and send_data
MSG_LENGTH_NUMBYTES = 7
MASTER_COMMAND_LENGTH = 6


class CommandParser:
    def __init__(self):
        self.tmp_buffer = bytearray(MSG_LENGTH_NUMBYTES)
        self.tmp_view = memoryview(self.tmp_buffer)

        self.data_buf = bytearray(9999999)
        self.data_view = memoryview(self.data_buf)

    def read_command(self, request):
        utils.recv_data_into(
            request,
            self.tmp_view[:MASTER_COMMAND_LENGTH],
            MASTER_COMMAND_LENGTH,
        )

        # rstrip is usefull when using nc
        cmd = self.tmp_buffer[:MASTER_COMMAND_LENGTH].decode("ascii")

        return cmd

    def read_data(self, request):
        # Read the num bytes of the data
        utils.recv_data_into(request, self.tmp_view, MSG_LENGTH_NUMBYTES)
        msg_length = int(self.tmp_buffer.decode("ascii"))
        print(f"Mesg length {msg_length}")

        # Read the message
        utils.recv_data_into(request, self.data_view[:msg_length], msg_length)

        return self.data_buf[:msg_length]


def read_command(request):
    parser = CommandParser()
    cmd = parser.read_command(request)
    # Possibly remove the padding value
    return cmd.rstrip()


def read_data(request):
    parser = CommandParser()
    return parser.read_data(request)


def send_command(request, cmd):
    reply = bytes(f"{cmd:6s}", "ascii")
    utils.send_data(request, reply)


def send_data(request, msg):
    reply = bytes(f"{len(msg):07}", "ascii")
    utils.send_data(request, reply)
    utils.send_data(request, msg)


class Client:
    def __init__(self, hostname, port):
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
            ClientStates.QUIT: self.quit,
            ClientStates.SELECT: self.select,
        }

    def get_list(self):
        send_command(self.sock, "list")
        cmd = read_command(self.sock)
        if cmd != "list":
            raise RuntimeError(f"Unexpected server reply, got {cmd}, expected 'list'")
        model_list = read_data(self.sock).decode("ascii").split("\n")
        print(model_list)

        self.selected_model, cancel = self.whiptail.menu(
            "Select the model you want to run. Cancel to quit", model_list
        )
        if cancel:
            return ClientStates.QUIT
        else:
            return ClientStates.SELECT

    def select(self):
        # Send to the server the selected model
        msg = bytes(self.selected_model, "ascii")

        send_command(self.sock, "slct")
        send_data(self.sock, msg)

        # TODO: wait until the server is ready to get and process
        # incoming data
        cmd = read_command(self.sock)
        print(f"Received : {cmd}")

        return ClientStates.INIT

    def quit(self):
        # request = bytes("quit", "ascii")
        # utils.send_data(self.sock, request)
        send_command(self.sock, "quit")
        self.keep_running = False
        return ClientStates.FINAL

    def run(self):
        while self.keep_running:
            self.current_state = self.callbacks[self.current_state]()
