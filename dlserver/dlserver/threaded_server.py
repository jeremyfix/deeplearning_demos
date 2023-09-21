# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import socketserver
import logging

# Local imports
from dlserver.handler_fsm import MasterStateMachine


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server):
        self.fsm = MasterStateMachine(server.models)
        super().__init__(request, client_address, server)

    def handle(self):
        while self.fsm.keeps_running:
            # command = str(self.request.recv(1024), "ascii")
            # logging.info(f"Handling {command} from {self.client_address}")
            self.fsm.step(self.request)
        logging.debug("Handler finished the transaction")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, port, config):
        super().__init__(
            server_address=("localhost", port),
            RequestHandlerClass=ThreadedTCPRequestHandler,
        )
        self.models = config
        print(self.models)
