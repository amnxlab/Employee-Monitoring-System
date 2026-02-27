"""
Watchdog auto-restart daemon.

Monitors the main monitoring system process. If it crashes or is killed
(e.g. from Task Manager), the watchdog automatically restarts it.

Usage:
    python watchdog.py

To register as a Windows Scheduled Task (runs at boot):
    schtasks /create /tn "MonitoringWatchdog" /tr "python <path>/watchdog.py" /sc onlogon /rl highest
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

LOG_FILE = Path(__file__).parent / "watchdog.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - WATCHDOG - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ]
)
logger = logging.getLogger("watchdog")

# Path to main.py
MAIN_SCRIPT = Path(__file__).parent / "main.py"
PYTHON_EXE = sys.executable

# Restart delay to avoid rapid-restart loops
RESTART_DELAY = 5        # seconds to wait before restarting
MAX_RAPID_RESTARTS = 5   # max restarts within RAPID_WINDOW
RAPID_WINDOW = 60        # seconds


def run_watchdog():
    """Main watchdog loop – restarts main.py if it exits."""
    logger.info(f"Watchdog started. Monitoring: {MAIN_SCRIPT}")
    logger.info(f"Python: {PYTHON_EXE}")

    restart_times = []

    while True:
        # Check for rapid restart loop
        now = time.time()
        restart_times = [t for t in restart_times if now - t < RAPID_WINDOW]
        if len(restart_times) >= MAX_RAPID_RESTARTS:
            logger.error(
                f"Rapid restart detected ({MAX_RAPID_RESTARTS} restarts in "
                f"{RAPID_WINDOW}s). Waiting 5 minutes before retrying..."
            )
            time.sleep(300)
            restart_times.clear()

        logger.info("Starting monitoring system...")
        try:
            proc = subprocess.Popen(
                [PYTHON_EXE, str(MAIN_SCRIPT)],
                cwd=str(MAIN_SCRIPT.parent),
            )
            restart_times.append(time.time())
            exit_code = proc.wait()
            logger.warning(f"Monitoring system exited with code {exit_code}")

            # Exit code 0 = clean shutdown (password-confirmed quit)
            if exit_code == 0:
                logger.info("Clean shutdown detected (password confirmed). Watchdog exiting.")
                break

        except KeyboardInterrupt:
            logger.info("Watchdog interrupted by user")
            if proc and proc.poll() is None:
                proc.terminate()
            break
        except Exception as e:
            logger.error(f"Error running main process: {e}")

        logger.info(f"Restarting in {RESTART_DELAY} seconds...")
        # Send Discord admin alert about the restart
        try:
            from integrations import DiscordNotifier
            DiscordNotifier().send_watchdog_restart(exit_code)
        except Exception as de:
            logger.debug(f"Could not send Discord restart alert: {de}")
        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    run_watchdog()
