"""
tools/system_tools.py — System control tools for Jarvis.

Covers: volume, brightness, lock, shutdown, restart, sleep, battery, Wi-Fi.

Platform notes:
  - Volume/brightness: Windows uses pycaw / screen-brightness-control; fallback to nircmd.
  - Power controls: OS-native commands (shutdown /s, pmset, systemctl).
  - Battery/Wi-Fi: psutil + subprocess.
"""

import subprocess
import sys
from langchain_core.tools import tool


# ── Volume ─────────────────────────────────────────────────────────────────────

@tool
def set_volume(level: int) -> str:
    """
    Set the system volume to a specific level (0–100).

    Args:
        level: Volume percentage between 0 (mute) and 100 (maximum).
    """
    level = max(0, min(100, level))
    try:
        if sys.platform == "win32":
            # pycaw is the cleanest Windows audio API
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            # pycaw uses a scalar 0.0–1.0
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        elif sys.platform == "darwin":
            subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=True)
        else:
            # ALSA / PulseAudio
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{level}%"], check=True)
        return f"Volume set to {level}%."
    except ImportError:
        return "pycaw is not installed. Run: pip install pycaw (Windows only)"
    except Exception as exc:
        return f"Could not set volume: {exc}"


# ── Brightness ─────────────────────────────────────────────────────────────────

@tool
def set_brightness(level: int) -> str:
    """
    Set the screen brightness to a specific level (0–100).

    Args:
        level: Brightness percentage between 0 (darkest) and 100 (brightest).
    """
    level = max(0, min(100, level))
    try:
        import screen_brightness_control as sbc  # pip install screen-brightness-control
        sbc.set_brightness(level)
        return f"Brightness set to {level}%."
    except ImportError:
        return "screen-brightness-control not installed. Run: pip install screen-brightness-control"
    except Exception as exc:
        return f"Could not set brightness: {exc}"


# ── Power controls ─────────────────────────────────────────────────────────────

@tool
def lock_windows() -> str:
    """
    Lock the current Windows session (equivalent to Win+L).
    """
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.LockWorkStation()
            return "Workstation locked."
        elif sys.platform == "darwin":
            subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to keystroke "q" using {command down, control down}'],
                check=True,
            )
            return "Screen locked."
        else:
            subprocess.run(["loginctl", "lock-session"], check=True)
            return "Session locked."
    except Exception as exc:
        return f"Could not lock: {exc}"


@tool
def shutdown_computer(delay_seconds: int = 60) -> str:
    """
    Schedule a system shutdown.

    Args:
        delay_seconds: Seconds to wait before shutting down. Default is 60.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(["shutdown", "/s", f"/t {delay_seconds}"], check=True)
        elif sys.platform == "darwin":
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay_seconds // 60}"], check=True)
        else:
            subprocess.run(["shutdown", "-h", f"+{delay_seconds // 60}"], check=True)
        return f"System will shut down in {delay_seconds} seconds."
    except Exception as exc:
        return f"Could not schedule shutdown: {exc}"


@tool
def restart_computer(delay_seconds: int = 60) -> str:
    """
    Schedule a system restart.

    Args:
        delay_seconds: Seconds to wait before restarting. Default is 60.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(["shutdown", "/r", f"/t {delay_seconds}"], check=True)
        elif sys.platform == "darwin":
            subprocess.run(["sudo", "shutdown", "-r", f"+{delay_seconds // 60}"], check=True)
        else:
            subprocess.run(["shutdown", "-r", f"+{delay_seconds // 60}"], check=True)
        return f"System will restart in {delay_seconds} seconds."
    except Exception as exc:
        return f"Could not schedule restart: {exc}"


@tool
def sleep_computer() -> str:
    """
    Put the computer to sleep immediately.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)
        elif sys.platform == "darwin":
            subprocess.run(["pmset", "sleepnow"], check=True)
        else:
            subprocess.run(["systemctl", "suspend"], check=True)
        return "Putting computer to sleep."
    except Exception as exc:
        return f"Could not sleep: {exc}"


# ── Battery ────────────────────────────────────────────────────────────────────

@tool
def get_battery_info() -> str:
    """
    Return current battery status including charge level and charging state.
    """
    try:
        import psutil  # pip install psutil
        battery = psutil.sensors_battery()
        if battery is None:
            return "No battery detected (desktop or unsupported platform)."
        status = "charging" if battery.power_plugged else "discharging"
        time_left = ""
        if battery.secsleft > 0 and not battery.power_plugged:
            h, m = divmod(battery.secsleft // 60, 60)
            time_left = f" — approximately {h}h {m}m remaining"
        return f"Battery: {battery.percent:.0f}% ({status}){time_left}."
    except ImportError:
        return "psutil not installed. Run: pip install psutil"
    except Exception as exc:
        return f"Could not read battery info: {exc}"


# ── Wi-Fi ──────────────────────────────────────────────────────────────────────

@tool
def get_wifi_info() -> str:
    """
    Return information about the current Wi-Fi connection (SSID, signal, IP).
    """
    try:
        import psutil  # pip install psutil
        import socket

        # Get local IP
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        # Network interface stats
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()

        wifi_info = []
        for iface, stat in stats.items():
            if stat.isup and iface.lower() not in ("lo", "loopback"):
                for addr in addrs.get(iface, []):
                    if addr.family == 2:  # AF_INET (IPv4)
                        wifi_info.append(f"Interface: {iface}, IP: {addr.address}")

        if sys.platform == "win32":
            result = subprocess.run(
                ["netsh", "wlan", "show", "interfaces"],
                capture_output=True, text=True,
            )
            # Extract SSID line
            for line in result.stdout.splitlines():
                if "SSID" in line and "BSSID" not in line:
                    wifi_info.insert(0, line.strip())
                    break
        elif sys.platform == "darwin":
            result = subprocess.run(
                ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"],
                capture_output=True, text=True,
            )
            for line in result.stdout.splitlines():
                if " SSID:" in line or "agrCtlRSSI:" in line:
                    wifi_info.insert(0, line.strip())

        return "\n".join(wifi_info) if wifi_info else "No active network connections found."
    except ImportError:
        return "psutil not installed. Run: pip install psutil"
    except Exception as exc:
        return f"Could not get Wi-Fi info: {exc}"
