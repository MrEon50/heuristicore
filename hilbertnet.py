#!/usr/bin/env python3
"""
HilbertNet: Uniwersalny framework — przestrzenna sieć neuronowa + logika binarna
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zastosowania:
  • ML binarny (training, inference)
  • Automaty logiczne (FSM, reguły)
  • Self-learning (wzmacnianie, discovery)
  • Konsola interaktywna / CLI
  • Integracja z projektami (API)
  • Automatyzacja programistyczna
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import struct
import pickle
import json
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import mmap
import os
import time
from abc import ABC, abstractmethod


# ============================================================================
# 1. HILBERT CURVE: transformacja 32-bit ↔ (x, y)
# ============================================================================

class HilbertMapping:
    """Mapuje 32-bit klucz ↔ (x, y) w przestrzeni Hilberta."""
    
    def __init__(self, order: int = 16):
        self.order = order
        self.size = 2 ** order
    
    @staticmethod
    def _rotate(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
        if ry == 0:
            if rx == 1:
                x, y = n - 1 - x, n - 1 - y
            x, y = y, x
        return x, y
    
    def index_to_xy(self, index: int) -> Tuple[int, int]:
        high = (index >> 16) & 0xFFFF
        low = index & 0xFFFF
        d = high
        x = y = 0
        s = 1
        while s < self.size:
            rx = 1 & (d >> 1)
            ry = 1 & (d ^ rx)
            x, y = self._rotate(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            d >>= 2
            s <<= 1
        return x, y
    
    def xy_to_index(self, x: int, y: int) -> int:
        d = 0
        s = self.size >> 1
        while s > 0:
            rx = int((x & s) > 0)
            ry = int((y & s) > 0)
            d += s * s * ((3 * rx) ^ ry)
            x, y = self._rotate(s, x, y, rx, ry)
            s >>= 1
        return d


# ============================================================================
# 2. REGUŁY: Plugin system dla operacji bitowych i logicznych
# ============================================================================

@dataclass
class RuleDefinition:
    """Definicja reguły."""
    name: str
    func: Callable
    params: Dict[str, Any] = None
    trainable: bool = False


class RuleRegistry:
    """Rejestr globalny reguł — łatwo dodawać własne."""
    
    def __init__(self):
        self.rules: Dict[str, RuleDefinition] = {}
        self._register_builtins()
    
    def _register_builtins(self):
        """Wbudowane reguły."""
        self.register('xor', lambda v, p: v ^ p, trainable=True)
        self.register('and', lambda v, p: v & p, trainable=False)
        self.register('or', lambda v, p: v | p, trainable=False)
        self.register('not', lambda v, _: 255 - v, trainable=False)
        self.register('rotate_l', lambda v, p: ((v << (p % 8)) | (v >> (8 - (p % 8)))) & 0xFF, trainable=True)
        self.register('rotate_r', lambda v, p: ((v >> (p % 8)) | (v << (8 - (p % 8)))) & 0xFF, trainable=True)
        self.register('add', lambda v, p: min(255, v + p), trainable=True)
        self.register('sub', lambda v, p: max(0, v - p), trainable=True)
        self.register('mul', lambda v, p: (v * p) % 256, trainable=True)
        self.register('hamming', lambda v, p: bin(v ^ p).count('1'), trainable=False)
        self.register('shift_l', lambda v, p: (v << (p % 8)) & 0xFF, trainable=True)
        self.register('shift_r', lambda v, p: v >> (p % 8), trainable=True)
        self.register('threshold', lambda v, p: 255 if v >= p else 0, trainable=False)
    
    def register(self, name: str, func: Callable, trainable: bool = False):
        """Rejestruj nową regułę."""
        self.rules[name] = RuleDefinition(name, func, trainable=trainable)
    
    def get(self, name: str) -> Callable:
        if name not in self.rules:
            raise ValueError(f"Unknown rule: {name}")
        return self.rules[name].func
    
    def list_rules(self) -> List[str]:
        return list(self.rules.keys())


# ============================================================================
# 3. PAMIĘĆ: Storage layer (Sparse/Full)
# ============================================================================

@dataclass
class NodeState:
    """Stan węzła."""
    value: int  # 0-255
    weight: float = 1.0
    activation: float = 0.0
    timestamp: int = 0
    flags: int = 0


class Storage:
    """Abstrakcyjna warstwa pamięci."""
    
    def read(self, key: int) -> int:
        raise NotImplementedError
    
    def write(self, key: int, value: int):
        raise NotImplementedError
    
    def get_state(self, key: int) -> NodeState:
        raise NotImplementedError
    
    def set_state(self, key: int, state: NodeState):
        raise NotImplementedError


class SparseStorage(Storage):
    """Pamięć rzadka (hashmap)."""
    
    def __init__(self):
        self.nodes: Dict[int, NodeState] = {}
    
    def read(self, key: int) -> int:
        return self.nodes.get(key, NodeState(0)).value
    
    def write(self, key: int, value: int):
        value = max(0, min(255, value))
        if key not in self.nodes:
            self.nodes[key] = NodeState(value)
        else:
            self.nodes[key].value = value
        self.nodes[key].timestamp = int(time.time())
    
    def get_state(self, key: int) -> NodeState:
        return self.nodes.get(key, NodeState(0))
    
    def set_state(self, key: int, state: NodeState):
        self.nodes[key] = state
    
    def keys(self):
        return list(self.nodes.keys())
    
    def to_dict(self) -> Dict:
        return {k: asdict(v) for k, v in self.nodes.items()}
    
    def from_dict(self, d: Dict):
        self.nodes = {int(k): NodeState(**v) for k, v in d.items()}


class FullStorage(Storage):
    """Pamięć pełna (mmap, 4 GiB)."""
    
    def __init__(self, filepath: str = '/tmp/hilbert_net.bin'):
        self.filepath = filepath
        self.file_size = 2**32
        self._init_file()
    
    def _init_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'wb') as f:
                f.write(b'\x00')
                f.seek(self.file_size - 1)
                f.write(b'\x00')
        self.file = open(self.filepath, 'r+b')
        self.mmap_data = mmap.mmap(self.file.fileno(), self.file_size)
    
    def read(self, key: int) -> int:
        return self.mmap_data[key % self.file_size]
    
    def write(self, key: int, value: int):
        value = max(0, min(255, value))
        self.mmap_data[key % self.file_size] = value
    
    def get_state(self, key: int) -> NodeState:
        return NodeState(self.read(key))
    
    def set_state(self, key: int, state: NodeState):
        self.write(key, state.value)
    
    def close(self):
        if hasattr(self, 'mmap_data'):
            self.mmap_data.close()
            self.file.close()
    
    def __del__(self):
        self.close()


# ============================================================================
# 4. SIEĆ: Węzły, połączenia, operacje
# ============================================================================

class HilbertNet:
    """Główna klasa sieci."""
    
    def __init__(self, mode: str = 'sparse', storage_path: Optional[str] = None,
                 order: int = 16):
        """
        mode: 'sparse' lub 'full'
        storage_path: ścieżka do mmap (jeśli full)
        order: rozmiar Hilberta (16 = 2^16 x 2^16)
        """
        self.hilbert = HilbertMapping(order=order)
        self.storage: Storage = SparseStorage() if mode == 'sparse' else FullStorage(storage_path or '/tmp/hnet.bin')
        self.rules = RuleRegistry()
        
        # Metadane sieci
        self.name: str = "HilbertNet"
        self.version: str = "1.0"
        self.metadata: Dict = {}
        
        # Historia zmian (dla treningowania)
        self.history: List[Dict] = []
        self.learning_rate: float = 0.01
        self.reward_history: List[float] = []
    
    # ========== OPERACJE PODSTAWOWE ==========
    
    def set(self, key: int, value: int, weight: float = 1.0):
        """Ustaw wartość węzła."""
        state = NodeState(value=value, weight=weight, timestamp=int(time.time()))
        self.storage.set_state(key, state)
    
    def get(self, key: int) -> int:
        """Pobierz wartość węzła."""
        return self.storage.read(key)
    
    def get_state(self, key: int) -> NodeState:
        """Pobierz pełny stan węzła."""
        return self.storage.get_state(key)
    
    def pos(self, key: int) -> Tuple[int, int]:
        """Zwróć pozycję (x, y) węzła w Hilberce."""
        return self.hilbert.index_to_xy(key)
    
    # ========== OPERACJE SIECIOWE ==========
    
    def neighbors(self, key: int, radius: int = 20) -> List[int]:
        """Zwróć sąsiadów w promieniu (Hilbert distance)."""
        x, y = self.hilbert.index_to_xy(key)
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.hilbert.size and 0 <= ny < self.hilbert.size:
                    neighbor_key = self.hilbert.xy_to_index(nx, ny)
                    neighbors.append(neighbor_key)
        return neighbors
    
    def propagate(self, source_key: int, rule: str = 'xor', param: int = 0xFF,
                  depth: int = 2, decay: float = 1.0) -> Dict[int, int]:
        """
        Propaguj sygnał z węzła źródłowego przez sieć.
        
        Args:
            source_key: węzeł źródłowy
            rule: nazwa reguły
            param: parametr reguły
            depth: głębokość propagacji
            decay: współczynnik zaniku (0-1)
        
        Returns:
            {węzeł: nowa_wartość}
        """
        rule_func = self.rules.get(rule)
        visited = set()
        queue = [(source_key, 0)]
        results = {}
        
        while queue:
            key, d = queue.pop(0)
            if key in visited or d > depth:
                continue
            visited.add(key)
            
            val = self.get(key)
            new_val = int(rule_func(val, param) * (decay ** d))
            new_val = max(0, min(255, new_val))
            self.set(key, new_val)
            results[key] = new_val
            
            if d < depth:
                for nkey in self.neighbors(key, radius=5):
                    if nkey not in visited:
                        queue.append((nkey, d + 1))
        
        return results
    
    def classify_region(self, key: int, radius: int = 50) -> Dict[str, float]:
        """Analizuj statystyki regionu wokół węzła."""
        nkeys = self.neighbors(key, radius=radius)
        values = [self.get(k) for k in nkeys]
        
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        values = np.array(values)
        return {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': int(values.min()),
            'max': int(values.max()),
            'count': len(values),
            'entropy': float(-np.sum(values/values.sum() * np.log2(values/values.sum() + 1e-10)))
        }
    
    # ========== LOGIKA ==========
    
    def apply_rule(self, key: int, rule: str, param: int = 0) -> int:
        """Zastosuj regułę do węzła."""
        val = self.get(key)
        rule_func = self.rules.get(rule)
        result = rule_func(val, param)
        self.set(key, int(result))
        return int(result)
    
    def apply_conditional(self, condition: Callable, action: Callable,
                          keys: Optional[List[int]] = None) -> int:
        """
        Warunkowe zastosowanie akcji.
        
        Args:
            condition: λ(value, state) -> bool
            action: λ(value, state) -> new_value
            keys: jeśli None, to wszystkie węzły
        """
        if keys is None:
            if isinstance(self.storage, SparseStorage):
                keys = self.storage.keys()
            else:
                return 0
        
        count = 0
        for k in keys:
            state = self.get_state(k)
            if condition(state.value, state):
                new_val = action(state.value, state)
                self.set(k, int(new_val))
                count += 1
        
        return count
    
    # ========== TRENING ==========
    
    def learn_pattern(self, pattern: Dict[int, int], label: int,
                      learning_rate: float = 0.01) -> float:
        """
        Ucz sieć rozpoznawania wzoru (supervised learning).
        
        pattern: {klucz: wartość}
        label: oczekiwana klasyfikacja (0 lub 1)
        """
        # Koduj wzór
        for key, val in pattern.items():
            self.set(key, val)
        
        # Forward: zbierz aktywacje
        activations = []
        for key in pattern.keys():
            region = self.classify_region(key, radius=10)
            activations.append(region['mean'])
        
        # Prosty neuron
        prediction = sum(activations) / (len(activations) + 1e-10) > 127
        error = float(label) - float(prediction)
        
        # Update (prosty SGD)
        for key in pattern.keys():
            val = self.get(key)
            delta = int(error * learning_rate * 255)
            self.set(key, max(0, min(255, val + delta)))
        
        loss = error ** 2
        self.history.append({'pattern': str(pattern)[:50], 'loss': loss, 'pred': prediction})
        self.reward_history.append(1 - loss)
        
        return loss
    
    def train_batch(self, dataset: List[Tuple[Dict, int]], epochs: int = 10) -> List[float]:
        """Ucz na batchu danych."""
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for pattern, label in dataset:
                loss = self.learn_pattern(pattern, label, self.learning_rate)
                epoch_loss += loss
            avg_loss = epoch_loss / len(dataset)
            losses.append(avg_loss)
        return losses
    
    def reinforce(self, action: Callable, reward: float, decay: float = 0.95):
        """Wzmacnianie: zapamiętaj akcję z nagrodą."""
        self.reward_history.append(reward)
        # Prosta heurystyka: propaguj nagrodę jako binarny sygnał
        if reward > 0.5:
            self.propagate(0, rule='add', param=int(reward * 50), depth=2, decay=decay)
    
    # ========== SERIALIZATION ==========
    
    def save(self, filepath: str):
        """Zapisz sieć do pliku."""
        data = {
            'name': self.name,
            'version': self.version,
            'learning_rate': self.learning_rate,
            'metadata': self.metadata,
            'history': self.history[-1000:],  # ostatnie 1000 wpisów
            'reward_history': self.reward_history[-1000:],
        }
        if isinstance(self.storage, SparseStorage):
            data['storage'] = self.storage.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Załaduj sieć z pliku."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.name = data.get('name', 'HilbertNet')
        self.learning_rate = data.get('learning_rate', 0.01)
        self.metadata = data.get('metadata', {})
        self.history = data.get('history', [])
        self.reward_history = data.get('reward_history', [])
        
        if 'storage' in data and isinstance(self.storage, SparseStorage):
            self.storage.from_dict(data['storage'])
    
    # ========== INFO ==========
    
    def info(self) -> Dict:
        """Zwróć informacje o sieci."""
        if isinstance(self.storage, SparseStorage):
            size = len(self.storage.nodes)
        else:
            size = "full (4GB)"
        
        return {
            'name': self.name,
            'version': self.version,
            'mode': 'sparse' if isinstance(self.storage, SparseStorage) else 'full',
            'nodes_active': size,
            'hilbert_order': self.hilbert.order,
            'rules': self.rules.list_rules(),
            'learning_rate': self.learning_rate,
            'history_len': len(self.history),
            'reward_mean': np.mean(self.reward_history) if self.reward_history else 0,
        }


# ============================================================================
# 5. INTERAKTYWNA KONSOLA
# ============================================================================

class HilbertShell:
    """Interaktywna konsola dla HilbertNet."""
    
    def __init__(self, net: HilbertNet):
        self.net = net
        self.commands = {
            'set': self.cmd_set,
            'get': self.cmd_get,
            'pos': self.cmd_pos,
            'neighbors': self.cmd_neighbors,
            'prop': self.cmd_propagate,
            'classify': self.cmd_classify,
            'rule': self.cmd_apply_rule,
            'info': self.cmd_info,
            'rules': self.cmd_list_rules,
            'save': self.cmd_save,
            'load': self.cmd_load,
            'help': self.cmd_help,
            'exit': self.cmd_exit,
        }
    
    def cmd_set(self, args: List[str]):
        """set <key> <value> [weight]"""
        if len(args) < 2:
            print("Usage: set <key> <value> [weight]")
            return
        key, val = int(args[0]), int(args[1])
        weight = float(args[2]) if len(args) > 2 else 1.0
        self.net.set(key, val, weight)
        print(f"✓ Set key {key} = {val} (weight={weight})")
    
    def cmd_get(self, args: List[str]):
        """get <key>"""
        if not args:
            print("Usage: get <key>")
            return
        key = int(args[0])
        val = self.net.get(key)
        state = self.net.get_state(key)
        print(f"Key {key}: value={val}, weight={state.weight:.2f}, activation={state.activation:.2f}")
    
    def cmd_pos(self, args: List[str]):
        """pos <key>"""
        if not args:
            print("Usage: pos <key>")
            return
        key = int(args[0])
        x, y = self.net.pos(key)
        print(f"Key {key} → position ({x}, {y})")
    
    def cmd_neighbors(self, args: List[str]):
        """neighbors <key> [radius]"""
        if not args:
            print("Usage: neighbors <key> [radius]")
            return
        key = int(args[0])
        radius = int(args[1]) if len(args) > 1 else 20
        neigh = self.net.neighbors(key, radius=radius)
        print(f"Neighbors of {key} (radius={radius}): {len(neigh)} nodes")
        print(f"Sample: {neigh[:10]}")
    
    def cmd_propagate(self, args: List[str]):
        """prop <key> <rule> [param] [depth]"""
        if len(args) < 2:
            print("Usage: prop <key> <rule> [param] [depth]")
            return
        key, rule = int(args[0]), args[1]
        param = int(args[2]) if len(args) > 2 else 0xFF
        depth = int(args[3]) if len(args) > 3 else 2
        result = self.net.propagate(key, rule=rule, param=param, depth=depth)
        print(f"✓ Propagated {rule} from {key}: {len(result)} nodes affected")
    
    def cmd_classify(self, args: List[str]):
        """classify <key> [radius]"""
        if not args:
            print("Usage: classify <key> [radius]")
            return
        key = int(args[0])
        radius = int(args[1]) if len(args) > 1 else 50
        stats = self.net.classify_region(key, radius=radius)
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    
    def cmd_apply_rule(self, args: List[str]):
        """rule <key> <rule_name> [param]"""
        if len(args) < 2:
            print("Usage: rule <key> <rule_name> [param]")
            return
        key, rule = int(args[0]), args[1]
        param = int(args[2]) if len(args) > 2 else 0
        result = self.net.apply_rule(key, rule, param)
        print(f"✓ Applied {rule}: result = {result}")
    
    def cmd_list_rules(self, args: List[str]):
        """rules"""
        rules = self.net.rules.list_rules()
        print(f"Available rules ({len(rules)}):")
        for r in rules:
            print(f"  • {r}")
    
    def cmd_info(self, args: List[str]):
        """info"""
        info = self.net.info()
        for k, v in info.items():
            print(f"  {k}: {v}")
    
    def cmd_save(self, args: List[str]):
        """save [filename]"""
        filename = args[0] if args else 'hilbert_net.json'
        self.net.save(filename)
        print(f"✓ Saved to {filename}")
    
    def cmd_load(self, args: List[str]):
        """load <filename>"""
        if not args:
            print("Usage: load <filename>")
            return
        self.net.load(args[0])
        print(f"✓ Loaded from {args[0]}")
    
    def cmd_help(self, args: List[str]):
        """help"""
        print("Commands:")
        for cmd in sorted(self.commands.keys()):
            print(f"  {cmd}")
    
    def cmd_exit(self, args: List[str]):
        """exit"""
        print("Goodbye!")
        raise KeyboardInterrupt
    
    def run(self):
        """Uruchom shell."""
        print("╔══════════════════════════════════════════════════════════╗")
        print("║          HilbertNet Interactive Shell v1.0              ║")
        print("║  Type 'help' for commands, 'exit' to quit               ║")
        print("╚══════════════════════════════════════════════════════════╝")
        
        try:
            while True:
                try:
                    line = input("hilbert> ").strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    cmd = parts[0]
                    args = parts[1:]
                    
                    if cmd in self.commands:
                        self.commands[cmd](args)
                    else:
                        print(f"Unknown command: {cmd}")
                
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        except KeyboardInterrupt:
            pass


# ============================================================================
# 6. DEMO I PRZYKŁADY
# ============================================================================

def demo_basic():
    """Demo: podstawowe operacje."""
    print("\n" + "="*60)
    print("DEMO 1: Podstawowe operacje")
    print("="*60)
    
    net = HilbertNet(mode='sparse')
    
    # Set wartości
    net.set(100, 200)
    net.set(200, 150)
    net.set(300, 100)
    
    print(f"Value at 100: {net.get(100)}")
    print(f"Position of 100: {net.pos(100)}")
    
    # Neighbors
    neigh = net.neighbors(100, radius=10)
    print(f"Neighbors of 100: {len(neigh)} nodes")
    
    # Classify region
    stats = net.classify_region(100, radius=50)
    print(f"Region stats: mean={stats['mean']:.1f}, entropy={stats['entropy']:.2f}")


def demo_propagation():
    """Demo: propagacja sygnału."""
    print("\n" + "="*60)
    print("DEMO 2: Propagacja sygnału")
    print("="*60)
    
    net = HilbertNet(mode='sparse')
    
    # Inicjalizuj randomowo
    for i in range(0, 5000, 100):
        net.set(i, np.random.randint(50, 200))
    
    # Propaguj
    print("Propagating XOR(0xFF) from key 2500, depth=3...")
    results = net.propagate(2500, rule='xor', param=0xFF, depth=3, decay=0.9)
    print(f"✓ Affected {len(results)} nodes")


def demo_ml():
    """Demo: Machine learning."""
    print("\n" + "="*60)
    print("DEMO 3: Machine Learning (binarna klasyfikacja)")
    print("="*60)
    
    net = HilbertNet(mode='sparse')
    
    # Prosty dataset
    dataset = [
        ({100: 200, 101: 180, 102: 150}, 1),  # Klasa 1
        ({200: 50, 201: 40, 202: 60}, 0),     # Klasa 0
        ({300: 210, 301: 190, 302: 160}, 1),
        ({400: 45, 401: 55, 402: 50}, 0),
    ]
    
    print("Training for 20 epochs...")
    losses = net.train_batch(dataset, epochs=20)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"History: {net.history[:3]}")


def demo_automation():
    """Demo: Automatyzacja logiczna."""
    print("\n" + "="*60)
    print("DEMO 4: Automatyzacja logiczna")
    print("="*60)
    
    net = HilbertNet(mode='sparse')
    
    # Ustaw stan początkowy
    for i in range(100, 110):
        net.set(i, np.random.randint(0, 256))
    
    # Warunkowa akcja: jeśli value > 200, odejmij 50
    count = net.apply_conditional(
        condition=lambda v, s: v > 200,
        action=lambda v, s: v - 50,
        keys=list(range(100, 110))
    )
    print(f"✓ Applied conditional to {count} nodes")
    
    # Wynik
    for i in range(100, 110):
        print(f"  Key {i}: {net.get(i)}")


def demo_custom_rules():
    """Demo: Własne reguły."""
    print("\n" + "="*60)
    print("DEMO 5: Rejestracja własnych reguł")
    print("="*60)
    
    net = HilbertNet(mode='sparse')
    
    # Zarejestruj własną regułę
    net.rules.register(
        'square',
        lambda v, p: min(255, (v * v) // 256),
        trainable=True
    )
    
    net.rules.register(
        'abs_diff',
        lambda v, p: abs(v - p),
        trainable=False
    )
    
    net.set(50, 100)
    result1 = net.apply_rule(50, 'square')
    print(f"square(100) = {result1}")
    
    result2 = net.apply_rule(50, 'abs_diff', 50)
    print(f"abs_diff({result1}, 50) = {result2}")
    
    print(f"\nDostępne reguły: {net.rules.list_rules()}")


def demo_interactive_shell():
    """Demo: Interaktywna konsola."""
    print("\n" + "="*60)
    print("DEMO 6: Interaktywna konsola")
    print("="*60)
    
    net = HilbertNet(mode='sparse')
    net.name = "MyFirstNet"
    
    shell = HilbertShell(net)
    
    # Auto-run kilka komend
    print("\n[Auto-running demo commands...]\n")
    shell.cmd_set(['1000', '150', '1.5'])
    shell.cmd_get(['1000'])
    shell.cmd_pos(['1000'])
    shell.cmd_apply_rule(['1000', 'xor', '255'])
    shell.cmd_info([])
    
    print("\n[Full interactive mode - uncomment shell.run() to use]\n")
    # Odkomentuj poniżej aby uruchomić pełny shell:
    # shell.run()


def demo_persistence():
    """Demo: Zapis i wczytanie."""
    print("\n" + "="*60)
    print("DEMO 7: Persistence (Save/Load)")
    print("="*60)
    
    net = HilbertNet(mode='sparse')
    net.name = "TestNet"
    net.metadata = {'author': 'demo', 'version': '1.0'}
    
    # Dodaj dane
    for i in range(0, 1000, 100):
        net.set(i, np.random.randint(0, 256))
    
    # Ucz
    dataset = [
        ({100: 200}, 1),
        ({200: 50}, 0),
    ]
    net.train_batch(dataset, epochs=5)
    
    # Zapisz
    net.save('/tmp/test_hilbert_net.json')
    print("✓ Saved to /tmp/test_hilbert_net.json")
    
    # Załaduj
    net2 = HilbertNet(mode='sparse')
    net2.load('/tmp/test_hilbert_net.json')
    print("✓ Loaded")
    print(f"Loaded net name: {net2.name}")
    print(f"Loaded net metadata: {net2.metadata}")
    print(f"Data matches: {net2.get(100) == net.get(100)}")


def demo_reinforcement():
    """Demo: Wzmacnianie (Reinforcement Learning)."""
    print("\n" + "="*60)
    print("DEMO 8: Reinforcement Learning")
    print("="*60)
    
    net = HilbertNet(mode='sparse')
    
    # Symuluj akcje i nagrody
    for episode in range(10):
        action = np.random.choice(['xor', 'add', 'mul'])
        reward = np.random.random()
        net.reinforce(lambda: net.propagate(100, rule=action, param=50), reward, decay=0.95)
        print(f"Episode {episode}: action={action}, reward={reward:.2f}")
    
    print(f"\nMean reward: {np.mean(net.reward_history):.2f}")


def test_hilbert_correctness():
    """Test: Poprawność Hilberta."""
    print("\n" + "="*60)
    print("TEST: Hilbert Curve Correctness")
    print("="*60)
    
    h = HilbertMapping()
    errors = 0
    
    for _ in range(10000):
        idx = np.random.randint(0, 2**32)
        x, y = h.index_to_xy(idx)
        back = h.xy_to_index(x, y)
        if back != idx:
            errors += 1
    
    if errors == 0:
        print("✓ 10,000 round-trip tests PASSED")
    else:
        print(f"✗ {errors} errors in 10,000 tests")


def benchmark_operations():
    """Benchmark: wydajność operacji."""
    print("\n" + "="*60)
    print("BENCHMARK: Operation Performance")
    print("="*60)
    
    net = HilbertNet(mode='sparse')
    n = 10000
    
    # Benchmark: SET
    t0 = time.time()
    for i in range(n):
        net.set(i, np.random.randint(0, 256))
    t_set = time.time() - t0
    print(f"SET {n} keys: {t_set*1e6/n:.2f} µs/op ({n/t_set:.0f} ops/sec)")
    
    # Benchmark: GET
    t0 = time.time()
    for i in range(n):
        _ = net.get(i)
    t_get = time.time() - t0
    print(f"GET {n} keys: {t_get*1e6/n:.2f} µs/op ({n/t_get:.0f} ops/sec)")
    
    # Benchmark: Neighbors
    t0 = time.time()
    for i in range(100):
        _ = net.neighbors(np.random.randint(0, n), radius=20)
    t_neigh = time.time() - t0
    print(f"NEIGHBORS (100 queries): {t_neigh*1e3/100:.2f} ms/query")
    
    # Benchmark: Classify
    t0 = time.time()
    for i in range(100):
        _ = net.classify_region(np.random.randint(0, n), radius=50)
    t_class = time.time() - t0
    print(f"CLASSIFY (100 queries): {t_class*1e3/100:.2f} ms/query")


# ============================================================================
# 7. MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import sys
    
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║               HilbertNet v1.0 — Demo Suite              ║")
    print("║  Universal Spatial Neural Network + Logic Automation    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    demos = {
        '1': ('Basic Operations', demo_basic),
        '2': ('Signal Propagation', demo_propagation),
        '3': ('Machine Learning', demo_ml),
        '4': ('Logic Automation', demo_automation),
        '5': ('Custom Rules', demo_custom_rules),
        '6': ('Interactive Shell', demo_interactive_shell),
        '7': ('Persistence', demo_persistence),
        '8': ('Reinforcement Learning', demo_reinforcement),
        'test': ('Correctness Test', test_hilbert_correctness),
        'bench': ('Benchmark', benchmark_operations),
    }
    
    def run_all():
        for k in ['1', '2', '3', '4', '5', '6', '7', '8']:
            demos[k][1]()
        test_hilbert_correctness()
        benchmark_operations()
    
    demos['all'] = ('Run All', run_all)
    
    if len(sys.argv) > 1 and sys.argv[1] in demos:
        demos[sys.argv[1]][1]()
    else:
        print("Usage: python script.py [demo_id]\n")
        print("Available demos:")
        for k, (name, _) in demos.items():
            print(f"  {k:4s} - {name}")
        print("\nExamples:")
        print("  python script.py 1        # Run demo 1")
        print("  python script.py all      # Run all demos")
        print("  python script.py bench    # Run benchmark\n")
        
        # Uruchom domyślny demo
        print("Running default demo (basic operations)...\n")
        demo_basic()