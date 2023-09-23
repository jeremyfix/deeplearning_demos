#!/usr/bin/env python3
# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with dlserver. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import argparse
import logging
import sys
import threading

# Local imports
from dlserver.threaded_server import ThreadedTCPServer
from dlserver.config import load_config


def main():

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        default=6008,
        type=int,
        help="The port on which to listen" " to an incoming image",
        action="store",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="The config to load. If you wish to use a"
        "config provided by the deeplearning_demos "
        "package, use --config config://",
        action="store",
        default="config://default.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None:
        return

    # Builds up the server
    server = ThreadedTCPServer(args.port, config)
    logging.info(f"Setup and ready listening on port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
