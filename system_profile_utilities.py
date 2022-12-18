import platform
import socket
import re
from distro import LinuxDistribution
import uuid
import json
from datetime import datetime
from cpuinfo import get_cpu_info
import psutil
import logging
import warnings

MACOS_VERSION = {"12.04": "Monterey", "12.4": "Monterey", "12.03": "Monterey", "12.3": "Monterey", "12.02": "Monterey",
                 "12.2": "Monterey", "12.01": "Monterey", "12.1": "Monterey", "12.00": "Monterey", "12.0": "Monterey",
                 "11.20": "Big Sur", "11.19": "Big Sur", "11.18": "Big Sur", "11.17": "Big Sur", "11.16": "Big Sur",
                 "11.15": "Big Sur", "11.14": "Big Sur", "11.13": "Big Sur", "11.12": "Big Sur", "11.11": "Big Sur",
                 "11.10": "Big Sur", "11.09": "Big Sur", "11.9": "Big Sur", "11.08": "Big Sur", "11.8": "Big Sur",
                 "11.07": "Big Sur", "11.7": "Big Sur", "11.06": "Big Sur", "11.6": "Big Sur", "11.05": "Big Sur",
                 "11.5": "Big Sur", "11.04": "Big Sur", "11.4": "Big Sur", "11.03": "Big Sur", "11.3": "Big Sur",
                 "11.02": "Big Sur", "11.2": "Big Sur", "11.01": "Big Sur", "11.1": "Big Sur", "11.00": "Big Sur",
                 "11.0": "Big Sur", "10.20": "Catalina", "10.19": "Catalina", "10.18": "Catalina", "10.17": "Catalina",
                 "10.16": "Catalina", "10.15": "Catalina", "10.14": "Mojave", "10.13": "High Sierra", "10.12": "Sierra",
                 "10.11": "X El Capitan", "10.10": "X Yosemite", "10.09": "X Mavericks", "10.9": "X Mavericks",
                 "10.08": "X Mountain Lion", "10.8": "X Mountain Lion", "10.07": "X Lion", "10.7": "X Lion",
                 "10.06": "X Snow Leopard", "10.6": "X Snow Leopard", "10.05": "X Leopard", "10.5": "X Leopard",
                 "10.04": "X Tiger", "10.4": "X Tiger", "10.03": "X Panther", "10.3": "X Panther", "10.02": "X Jaguar",
                 "10.2": "X Jaguar", "10.01": "X Puma", "10.1": "X Puma", "10.0": "X Cheetah", "10.00": "X Cheetah"}

PLATFORM = platform.system()
if PLATFORM == "Windows":
    os_platform, os_platform_release, os_platform_version = platform.system(), platform.release(), platform.version()
elif PLATFORM == "Linux":
    os_platform, os_platform_version, os_platform_release = LinuxDistribution().linux_distribution(
        full_distribution_name=False)
elif PLATFORM == "Darwin":
    warnings.warn("MacOS is not fully supported, MacOS beyond Catalina might not be correct", stacklevel=2)
    os_platform, os_platform_version, os_platform_release, = "Darwin MacOS", MACOS_VERSION.get(platform.mac_ver()[0]), \
                                                             platform.mac_ver()[0]


def get_size(numbytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    :return: a string of size in proper format
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if numbytes < factor:
            return f"{numbytes:.2f}{unit}{suffix}"
        numbytes /= factor


def get_system_info():
    """
    Utility function to get system information
    :return: a dictionary of system information
    """
    try:
        info = dict()

        info["SYSTEM"] = dict(
            booted_up_time=str(datetime.fromtimestamp(psutil.boot_time())),
            hostname=socket.gethostname(),
            ip_address=socket.gethostbyname(socket.gethostname()),
            mac_address=':'.join(re.findall('..', '%012x' % uuid.getnode())),
            python_version=platform.python_version(),
            python_compiler=platform.python_compiler(),
            python_build=platform.python_build(),
            python_implementation=platform.python_implementation(),
        )

        info["OS"] = dict(
            platform=os_platform,
            platform_release=os_platform_release,
            platform_version=os_platform_version
        )
        # if info["OS"]["platform"] == "Linux":
        #     info["OS"]["linux_platform_info"] = LinuxDistribution.linux_distribution(full_distribution_name=True)

        cpu_info = get_cpu_info()
        info["CPU"] = dict(
            name=cpu_info['brand_raw'],
            bits=cpu_info['bits'],
            architechture=cpu_info['arch'],
            arch_string_raw=cpu_info['arch_string_raw'],
            cpu_physical_cores=psutil.cpu_count(logical=False),
            cpu_logical_cores=psutil.cpu_count(logical=True),
            cpu_max_freq_advertised=cpu_info['hz_advertised_friendly'],
            cpu_max_freq_actual=cpu_info['hz_actual_friendly'],
            cpu_min_freq_actual=cpu_info['hz_advertised'][1],
            l2_cache_size=get_size(cpu_info['l2_cache_size'], suffix='KB'),
            l3_cache_size=get_size(cpu_info['l3_cache_size'], suffix='KB'),
            vendor_id_raw=cpu_info['vendor_id_raw']
        )

        memory_info = psutil.virtual_memory()
        info["RAM"] = dict(
            ram_total=str(get_size(memory_info.total)),
            ram_available=str(get_size(memory_info.available)))

        swap_info = psutil.swap_memory()
        info["SWAP"] = dict(
            swap_total=str(get_size(swap_info.total)),
            swap_free=str(get_size(swap_info.free))
        )

        disk_info = psutil.disk_partitions()
        info['DISK'] = {"list disk": [(index, disk.device) for index, disk in enumerate(disk_info)]}
        for i, disk in enumerate(disk_info):
            info['DISK'][f'disk_{i}_total'] = str(get_size(psutil.disk_usage(disk.mountpoint).total))
            info['DISK'][f'disk_{i}_free'] = str(get_size(psutil.disk_usage(disk.mountpoint).free))

        return json.dumps(info)
    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    from pprint import pprint

    pprint(json.loads(get_system_info()))

    pprint(get_cpu_info())
