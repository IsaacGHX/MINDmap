import time
import pickle
import json
import msgpack
from pathlib import Path
import random
import string
import sys
import gc
import os
from sys import getsizeof
from collections import deque

"""
message pack 的速度是最最快的，原因可能是其大量的内存的占用
"""


class MemoryTracker:
    """内存使用跟踪器"""

    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0

    def start(self):
        """开始跟踪内存"""
        gc.collect()  # 强制垃圾回收
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory

    def stop(self):
        """停止跟踪并返回内存使用情况"""
        current = self._get_memory_usage()
        peak = self.peak_memory
        diff = current - self.start_memory
        return {
            'start': self.start_memory,
            'current': current,
            'peak': peak,
            'diff': diff
        }

    def update(self):
        """更新峰值内存使用"""
        current = self._get_memory_usage()
        self.peak_memory = max(self.peak_memory, current)

    def _get_memory_usage(self):
        """获取当前内存使用"""
        if hasattr(sys, 'getwindowsversion'):
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        else:
            # Linux/Unix系统
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class TreeNode:
    """优化的树节点类"""
    __slots__ = ['id', 'parent_id', 'value']

    def __init__(self, id, parent_id, value):
        self.id = id
        self.parent_id = parent_id
        self.value = value


def format_size(bytes):
    """格式化字节大小为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"


class BenchmarkSuite:
    def __init__(self, sizes=None):
        self.sizes = sizes or [10, 100, 1000, 10000, 100000]
        self.results = []
        self.memory_tracker = MemoryTracker()

    def generate_data(self, size):
        """生成测试数据"""
        data = {}
        chars = string.ascii_letters + string.digits
        for i in range(size):
            node = TreeNode(
                id=i,
                parent_id=random.randint(-1, i - 1) if i > 0 else -1,
                value=''.join(random.choices(chars, k=10))
            )
            data[i] = node
        return data

    def measure_operation(self, func, data):
        """测量操作的时间和内存使用"""
        self.memory_tracker.start()
        start_time = time.time()

        result = func(data)

        duration = time.time() - start_time
        memory_stats = self.memory_tracker.stop()

        return {
            'duration': duration,
            'memory': memory_stats,
            'result': result
        }

    def benchmark_format(self, data, format_name):
        """对特定格式进行基准测试"""

        def pickle_write(d):
            with open('test.pkl', 'wb') as f:
                pickle.dump(d, f)

        def pickle_read(d):
            with open('test.pkl', 'rb') as f:
                return pickle.load(f)

        def json_write(d):
            # 转换为可JSON序列化的格式
            json_data = {str(k): {'id': v.id, 'parent_id': v.parent_id, 'value': v.value}
                         for k, v in d.items()}
            with open('test.json', 'w') as f:
                json.dump(json_data, f)

        def json_read(d):
            with open('test.json', 'r') as f:
                return json.load(f)

        def msgpack_write(d):
            # 转换为可序列化的格式
            pack_data = {str(k): {'id': v.id, 'parent_id': v.parent_id, 'value': v.value}
                         for k, v in d.items()}
            with open('test.msgpack', 'wb') as f:
                msgpack.pack(pack_data, f)

        def msgpack_read(d):
            with open('test.msgpack', 'rb') as f:
                return msgpack.unpack(f)

        operations = {
            'pickle': (pickle_write, pickle_read),
            'json': (json_write, json_read),
            'msgpack': (msgpack_write, msgpack_read)
        }

        write_func, read_func = operations[format_name]

        # 测试写入
        write_stats = self.measure_operation(write_func, data)

        # 测试读取
        read_stats = self.measure_operation(read_func, data)

        # 获取文件大小
        file_extension = {'pickle': '.pkl', 'json': '.json', 'msgpack': '.msgpack'}
        file_size = os.path.getsize(f'test{file_extension[format_name]}')

        return {
            'write': write_stats,
            'read': read_stats,
            'file_size': file_size
        }

    def run(self):
        """运行完整的基准测试"""
        for size in self.sizes:
            print(f"\nTesting size: {size} nodes")
            data = self.generate_data(size)
            data_size = sum(getsizeof(node) for node in data.values())

            size_results = {
                'size': size,
                'data_size': data_size,
                'formats': {}
            }

            for format_name in ['pickle', 'json', 'msgpack']:
                print(f"Testing {format_name}...")
                try:
                    results = self.benchmark_format(data, format_name)
                    size_results['formats'][format_name] = results

                    # 打印当前结果
                    print(f"  Write: {results['write']['duration']:.4f}s "
                          f"(Memory: {format_size(results['write']['memory']['peak'])})")
                    print(f"  Read: {results['read']['duration']:.4f}s "
                          f"(Memory: {format_size(results['read']['memory']['peak'])})")
                    print(f"  File size: {format_size(results['file_size'])}")

                except Exception as e:
                    print(f"Error testing {format_name}: {str(e)}")
                    size_results['formats'][format_name] = None

                # 清理测试文件
                for ext in ['.pkl', '.json', '.msgpack']:
                    Path(f'test{ext}').unlink(missing_ok=True)

                gc.collect()  # 强制垃圾回收

            self.results.append(size_results)

        return self.format_results()

    def format_results(self):
        """格式化结果为可读格式"""
        summary = []
        for result in self.results:
            size = result['size']
            print(f"\nSummary for {size} nodes:")
            print(f"Original data size: {format_size(result['data_size'])}")

            for format_name, data in result['formats'].items():
                if data:
                    summary.append({
                        'Nodes': size,
                        'Format': format_name,
                        'Write Time': f"{data['write']['duration']:.4f}s",
                        'Write Memory': format_size(data['write']['memory']['peak']),
                        'Read Time': f"{data['read']['duration']:.4f}s",
                        'Read Memory': format_size(data['read']['memory']['peak']),
                        'File Size': format_size(data['file_size'])
                    })

        return summary


# 运行基准测试
if __name__ == "__main__":
    # 使用不同的数据规模进行测试
    # sizes = [10, 100, 1000, 10000, 100000, 1000000]
    sizes = [1000000]

    try:
        benchmark = BenchmarkSuite(sizes)
        results = benchmark.run()

        # 打印完整的比较表格
        print("\nComplete Benchmark Results:")
        headers = ['Nodes', 'Format', 'Write Time', 'Write Memory',
                   'Read Time', 'Read Memory', 'File Size']

        # 计算每列的最大宽度
        widths = {header: len(header) for header in headers}
        for result in results:
            for header in headers:
                width = len(str(result[header]))
                widths[header] = max(widths[header], width)

        # 打印表头
        header_line = " | ".join(
            header.ljust(widths[header]) for header in headers
        )
        print("\n" + header_line)
        print("-" * len(header_line))

        # 打印数据行
        for result in results:
            line = " | ".join(
                str(result[header]).ljust(widths[header])
                for header in headers
            )
            print(line)

    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        raise