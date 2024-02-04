import gc
import random
import socket
import string
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from platform import machine
from typing import Tuple

if machine().lower().startswith("arm"):
    from cyclops.cyclops import Cyclops
else:
    from hwcounter import Timer as Cyclops


@dataclass
class Cryterion:
    # Size of input data to the algorithm (bytes).
    data_size: int

    # Size of the key (bytes).
    key_size: int

    # Size of each block (bytes).
    block_size: int

    # Size of code (bytes).
    code_size: int

    # No. of clock cycles used during the algorithm execution.
    clock_cycles: int

    # Time taken during the algorithm execution (ns).
    duration: int

    # Memory usage during the algorithm execution (bytes).
    memory_usage: int

    @property
    def latency_hardware(self):
        return self.duration

    @property
    def latency_software(self) -> float:
        """The amount of clock cycles per block (during encryption)."""
        assert (
            self.data_size % self.block_size == 0
        ), "data_size must be a multiple of block_size"
        blocks = self.data_size // self.block_size
        return self.clock_cycles / blocks

    @property
    def throughput_hardware(self) -> float:
        """Data (plaintext/ciphertext) processed per time unit in Bps (bytes per second)."""
        return self.data_size * 1e9 / self.duration

    @property
    def throughput_software(self) -> float:
        """Data (plaintext/ciphertext) processed per clock cycle (bytes per clock cycle)."""
        return self.data_size / self.clock_cycles

    @property
    def power(self) -> float:
        """A value that is directly proportional to the power required for this algorithm.
        (power_factor * k) µW (Micro watts) where k is a constant factor that depends on the CPU.
        """
        SCALING_FACTOR: float = 1e-17
        # 1e9 -> ns to s, 1e6 -> W to µW, SCALING_FACTOR -> scaling clock_cycles to Joules.
        power_factor = self.clock_cycles * 1e9 * 1e6 * SCALING_FACTOR / self.duration
        return power_factor

    @property
    def energy(self) -> float:
        """Energy consumption in µJ (Micro Joules)."""
        return self.power * self.duration * 1e-9

    @property
    def energy_per_bit(self) -> float:
        """Energy consumption per bit in µJ (Micro Joules).
        A factor of k is assumed where k is a constant factor that depends on the CPU.
        """
        return self.latency_software * self.power / (self.block_size * 8)

    @property
    def efficiency_hardware(self) -> float:
        assert False, "NotImplemented"

    @property
    def efficiency_software(self) -> float:
        """Software Efficiency = Throughput[Bps] / CodeSize[B]
        The unit for software efficiency is s^-1 (seconds inverse).
        Here, code size is the algorithm size.
        """
        return self.throughput_hardware / self.code_size

    @property
    def security_level(self) -> float:
        """Security level of the algorithm in terms of bits."""
        return self.key_size * 8

    def __str__(self) -> str:
        return "\n".join(
            (
                f"Data (plaintext/ciphertext) Size: {self.data_size} bytes",
                f"Memory Usage: {self.memory_usage * 1e-3:.3f} kB",
                f"Time Taken: {self.duration * 1e-6:.3f} ms",
                f"Clock Cycles: {self.clock_cycles}",
                f"Code Size: {self.code_size} bytes",
                f"Hardware Latency: {self.latency_hardware * 1e-6:.3f} ms",
                f"Software Latency: {self.latency_software:.3f} clock cycles per block",
                f"Hardware Throughput: {self.throughput_hardware:.3f} Bps",
                f"Software Throughput: {self.throughput_software:.3e} bytes per clock cycle",
                f"Power: {self.power:.06f}k µW (k is a constant factor that's CPU dependent)",
                f"Energy Per Bit: {self.energy_per_bit:.06f}k µJ (k is a constant factor that's CPU dependent)",
                f"Software Efficiency: {self.efficiency_software} s^-1",
                f"Security Level: {self.security_level} bits",
            )
        )


def benchmark_fn(
    fn: Callable[[bytes], bytes],
    data: bytes,
    key_size: int,
    block_size: int,
    code_size: int,
) -> Tuple[bytes, Cryterion]:
    gc.collect()
    gc_old = gc.isenabled()
    gc.disable()

    tracemalloc.start()
    start_time = time.process_time_ns()

    with Cyclops() as cyclops:
        result = fn(data)

    duration = time.process_time_ns() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if gc_old is True:
        gc.enable()

    data_size = len(data)
    clock_cycles = cyclops.cycles
    benchmark = Cryterion(
        data_size, key_size, block_size, code_size, clock_cycles, duration, peak
    )
    return result, benchmark


def encrypt(
    fn: Callable[[bytes], bytes],
    data: bytes,
    key_size: int,
    block_size: int,
    code_size: int,
) -> bytes:
    result, benchmark = benchmark_fn(fn, data, key_size, block_size, code_size)
    print(benchmark)
    return result


def decrypt(
    fn: Callable[[bytes], bytes],
    data: bytes,
    key_size: int,
    block_size: int,
    code_size: int,
) -> bytes:
    result, benchmark = benchmark_fn(fn, data, key_size, block_size, code_size)
    print(benchmark)
    return result


def random_bytes(length: int) -> bytes:
    return random.randbytes(length)


def random_text(length: int) -> bytes:
    charset = string.printable[:-3].encode("ascii")
    return bytes(random.choices(charset, k=length))


def pad(data_to_pad: bytes, block_size: int) -> bytes:
    """The padding scheme is `data + 0xFF + 0x00 bytes till the length is a multiple of block_size`
    Reference: Crypto.Util.Padding
    """
    assert block_size != 0
    return b"".join(
        (data_to_pad, b"\xff", -(len(data_to_pad) + 1) % block_size * b"\x00")
    )


def unpad(padded_data: bytes) -> bytes:
    idx = padded_data.rfind(b"\xff")
    assert idx != -1
    return padded_data[:idx]


def sendall(data: bytes, host: str, port=8000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(data)
    s.close()


def recvall(host: str, port=8000, max_bufsize=100 * 1024) -> bytes:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen()
    conn, addr = s.accept()

    # print(f"Connected to {addr}")

    received = b""

    while True:
        data = conn.recv(4096)
        received += data
        if len(data) == 0 or len(received) >= max_bufsize:
            break

    conn.close()
    s.close()

    return received


def code_size_from_files(files: list[str]) -> int:
    return sum(len(Path(f).read_bytes()) for f in files)


def int_from_bytes(b: bytes) -> int:
    return int.from_bytes(b, "big")


def int_to_bytes(x: int) -> bytes:
    length = (x.bit_length() + 7) // 8
    return x.to_bytes(length, "big")


if __name__ == "__main__":
    # Test `pad` and `unpad`.
    for block_size in range(1, 10):
        for _ in range(10_000):
            data = random.randbytes(random.randint(0, 64))
            assert unpad(pad(data, block_size)) == data
