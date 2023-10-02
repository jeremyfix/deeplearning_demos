# coding: utf-8

# This file is part of dlserver.

# dlserver is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# dlserver is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# dlserver. If not, see <https://www.gnu.org/licenses/>.

# External imports
import yaml
import os
import logging
import pathlib
from typing import Union

# Local imports


def load_config(config_path: Union[str, pathlib.Path]):
    # Loads the provided config
    if isinstance(config_path, str):
        if len(config_path) >= 9 and config_path[:9] == "config://":
            if len(config_path) == 9:
                # Check the available configs
                logging.info("Available configs : ")
                config_path = pathlib.Path(os.path.dirname(__file__)) / "configs/"
                logging.info(
                    "\n".join(
                        [f"- config://{c.name} {c}" for c in config_path.glob("*")]
                    )
                )
                return
            else:
                config_name = config_path[9:]
                config_path = pathlib.Path(os.path.dirname(__file__)) / "configs/"
                config_path = config_path / config_name
        else:
            config_path = pathlib.Path(config_path)

    # Load the config file
    if not config_path.exists():
        logging.error(f"Cannot find the requested file {config_path}")
        return

    logging.info(f"Loading {config_path}")
    return yaml.safe_load(open(str(config_path), "r"))
