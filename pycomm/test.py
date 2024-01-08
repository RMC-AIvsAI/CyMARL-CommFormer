from components.episode import PlayGame
import torch
from utils.dotdic import DotDic


episode = DotDic({})
episode.s_t = []
episode.r_t = []
episode.a_t = []

game = PlayGame(episode, "pycomm/games/game_info.txt")
