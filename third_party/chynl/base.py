# Vendored from chynl/snake (https://github.com/chynl/snake)
# SPDX-License-Identifier: MIT
# Full license text: third_party/chynl/LICENSE

from typing import Protocol

from src.direc import Direction
from src.snake import Snake


class Agent(Protocol):
    def next_direc(self, snake: Snake) -> Direction: ...
