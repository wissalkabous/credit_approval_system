"""
Implémentation du Logger - Respecte DIP
"""
from src.interfaces import ILogger
from datetime import datetime
from typing import Optional
from pathlib import Path


class ConsoleLogger(ILogger):
    """Logger qui affiche dans la console."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def info(self, message: str) -> None:
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] INFO  {message}")

    def error(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] ERROR {message}")

    def warning(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] WARN  {message}")


class FileLogger(ILogger):
    """Logger qui écrit dans un fichier."""
    
    def __init__(self, log_file: str = "logs/app.log"):
        self.log_file = log_file
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def info(self, message: str) -> None:
        self._write("INFO", message)
    
    def error(self, message: str) -> None:
        self._write("ERROR", message)
    
    def warning(self, message: str) -> None:
        self._write("WARNING", message)
    
    def _write(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {level}: {message}\n")


class HybridLogger(ILogger):
    """Logger qui écrit dans la console ET dans un fichier."""
    
    def __init__(self, log_file: str = "logs/app.log", verbose: bool = True):
        self.console_logger = ConsoleLogger(verbose)
        self.file_logger = FileLogger(log_file)
    
    def info(self, message: str) -> None:
        self.console_logger.info(message)
        self.file_logger.info(message)
    
    def error(self, message: str) -> None:
        self.console_logger.error(message)
        self.file_logger.error(message)
    
    def warning(self, message: str) -> None:
        self.console_logger.warning(message)
        self.file_logger.warning(message)
