"""SMILES Tokenizer（来自 util/tokens.py，未做实质修改）。"""
from __future__ import annotations
import os
import pickle
import torch

from ._config import load_config


class Tokenizer:
    """支持多字符 token、方括号原子、特殊 token 的 SMILES 分词器。"""

    def __init__(self, multiCharTokens, handleBraces=True,
                 toolTokens=('<start>', '<end>', '<pad>')):
        self.tokensDict, self.tokensInvDict = dict(), dict()
        self.toolTokens = toolTokens
        self.handleBraces = handleBraces
        self.multiCharTokens = [t.lower() for t in sorted(multiCharTokens, key=len, reverse=True)]
        for token in self.toolTokens:
            self.addToken(token)

    def addToken(self, token):
        if token not in self.tokensDict:
            num = len(self.tokensDict)
            self.tokensDict[token] = num
            self.tokensInvDict[num] = token

    def tokenize(self, smilesStrs, useTokenDict=False):
        vectors = []
        if useTokenDict:
            appliedMultiCharToken = sorted(
                [k for k in self.tokensDict.keys()
                 if k not in self.toolTokens and len(k) > 1],
                key=len, reverse=True)
        else:
            appliedMultiCharToken = self.multiCharTokens

        for smilesStr in smilesStrs:
            currentVector = []
            startIdx, length = 0, len(smilesStr)
            while startIdx < length:
                foundMultiCharToken = False
                if self.handleBraces and not useTokenDict and smilesStr[startIdx] == '[':
                    endIdx = smilesStr.index(']', startIdx) + 1
                    foundMultiCharToken = True
                else:
                    endIdx = startIdx + 1
                    for token in appliedMultiCharToken:
                        candidate_end = startIdx + len(token)
                        if candidate_end > length:
                            continue
                        seg = smilesStr[startIdx: candidate_end]
                        if (not useTokenDict and seg.lower() == token) or \
                           (useTokenDict and seg == token):
                            endIdx = candidate_end
                            foundMultiCharToken = True
                            break
                if not foundMultiCharToken:
                    endIdx = startIdx + 1
                currentVector.append(smilesStr[startIdx: endIdx])
                startIdx = endIdx
            vectors.append(currentVector)
        return vectors

    def getTokensSize(self) -> int:
        return len(self.tokensDict)

    def getTokensNum(self, token: str) -> int:
        return self.tokensDict[token]

    def getNumVector(self, vectors, addStart=False, addEnd=False):
        numVector = []
        for vec in vectors:
            currentVec = []
            if addStart:
                currentVec.append(self.tokensDict['<start>'])
            for elem in vec:
                currentVec.append(self.tokensDict[elem])
            if addEnd:
                currentVec.append(self.tokensDict['<end>'])
            numVector.append(currentVec)
        return numVector

    def getSmiles(self, numVectors):
        """去除 <start>/<end>/<pad> 工具 token，将索引序列还原为 SMILES。"""
        smilesList = []
        tool_ids = {self.tokensDict[t] for t in self.toolTokens}
        for vec in numVectors:
            if isinstance(vec, torch.Tensor):
                vec = vec.cpu().numpy().tolist()
            filtered = [int(n) for n in vec if int(n) not in tool_ids]
            smilesList.append(
                ''.join(self.tokensInvDict[n] for n in filtered if n in self.tokensInvDict))
        return smilesList


def _build_tokenizer(smi_file: str, handleBraces: bool = True) -> Tokenizer:
    tokenizer = Tokenizer(
        ['Li', 'Be', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ca',
         'As', 'Se', 'Br', 'Te', '@@'],
        handleBraces=handleBraces)
    with open(smi_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = set(*tokenizer.tokenize([line]))
            for tok in tokens:
                tokenizer.addToken(tok)
    return tokenizer


def get_tokenizer() -> Tokenizer:
    """加载（或首次构建）pcvae_tool 自带的 tokenizer。"""
    cfg = load_config()
    pkl_path = cfg['fname_tokenizer']
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    tokenizer = _build_tokenizer(cfg['token_file'])
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    return tokenizer
