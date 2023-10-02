#!/usr/bin/env python3
# coding: utf-8

# This file is part of dlclient.

# dlclient is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# dlclient is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# dlclient. If not, see <https://www.gnu.org/licenses/>.

# Standard imports
import logging
import sys
import argparse

# Local imports
from dlclient.client import Client


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname",
        default="localhost",
        type=str,
        help="The host to connect to",
        action="store",
    )
    parser.add_argument(
        "--port",
        default=6008,
        type=int,
        help="The port on which to connect",
        action="store",
    )
    args = parser.parse_args()

    client = Client(args.hostname, args.port)
    client.run()


if __name__ == "__main__":
    main()
