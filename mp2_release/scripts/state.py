from typing import List, Set
from collections import deque

class Token:
    def __init__(self, idx: int, word: str, pos: str):
        self.idx = idx # Unique index of the token
        self.word = word # Token string
        self.pos  = pos # Part of speech tag 

class DependencyEdge:
    def __init__(self, source: Token, target: Token, label:str):
        self.source = source  # Source token index
        self.target = target  # target token index
        self.label  = label  # dependency label
        pass


class ParseState:
    def __init__(self, stack: List[Token], parse_buffer: List[Token], dependencies: List[DependencyEdge]):
        self.stack = stack # A stack of token indices in the sentence. Assumption: the root token has index 0, the rest of the tokens in the sentence starts with 1.
        self.parse_buffer = deque(parse_buffer)  # A buffer of token indices
        self.dependencies = dependencies
        # print("init")
        # print("pb:", [a.word for a in self.parse_buffer])
        # print("s:", [a.word for a in self.stack])

    def add_dependency(self, source_token, target_token, label):
        self.dependencies.append(
            DependencyEdge(
                source=source_token,
                target=target_token,
                label=label,
            )
        )


def shift(state: ParseState) -> None:
    buffer_value = state.parse_buffer.popleft() 
    state.stack.append(buffer_value)
    # print("shift")
    # print("pb:", [a.word for a in state.parse_buffer])
    # print("s:", [a.word for a in state.stack])

def left_arc(state: ParseState, label: str) -> None:
    source = state.stack.pop()
    target = state.stack.pop()
    state.add_dependency(source, target, label)
    state.stack.append(source)
    # print("left_arc")
    # print("pb:", [a.word for a in state.parse_buffer])
    # print("s:", [a.word for a in state.stack])



def right_arc(state: ParseState, label: str) -> None:
    target = state.stack.pop()
    source = state.stack.pop()
    state.add_dependency(source, target, label)
    state.stack.append(source)
    # print("right_arc")
    # print("pb:", [a.word for a in state.parse_buffer])
    # print("s:", [a.word for a in state.stack])



def is_final_state(state: ParseState, cwindow: int) -> bool:
    # print("final")
    # print("pb:", [a.word for a in state.parse_buffer])
    # print("s:", [a.word for a in state.stack])
    buffer_words = [a.word for a in state.parse_buffer]
    stack_words = [a.word for a in state.stack]
    if "[NULL]" in buffer_words and "[NULL]" in stack_words:
        if len(state.parse_buffer) <= cwindow and len(state.stack) <= cwindow+1:
            return True
        return False
    else:
        if len(state.parse_buffer) == 0 and len(state.stack) == 1:
            return True
        return False
