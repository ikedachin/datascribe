#!/usr/bin/env python3


# ==============================
# カラー装飾用の定数
# ==============================
class Colors:
    RESET = '\033[0m' # Reset color
    BOLD = '\033[1m' # Bold text
    RED = '\033[91m' # Red text
    GREEN = '\033[92m' # Green text
    YELLOW = '\033[93m' # Yellow text
    BLUE = '\033[94m' # Blue text
    MAGENTA = '\033[95m' # Magenta text
    CYAN = '\033[96m' # Cyan text
    GRAY = '\033[90m' # Gray text

def msg_info(msg):
    return f"{Colors.BLUE}{Colors.BOLD}💡 [INFO]{Colors.RESET} {Colors.CYAN}{msg}{Colors.RESET}"

def msg_error(msg):
    return f"{Colors.RED}{Colors.BOLD}❌ [ERROR]{Colors.RESET} {Colors.RED}{msg}{Colors.RESET}"

def msg_debug(msg):
    return f"{Colors.YELLOW}🔍 [DEBUG]{Colors.RESET} {Colors.GRAY}{msg}{Colors.RESET}"

def msg_success(msg):
    return f"{Colors.GREEN}{Colors.BOLD}✅ [SUCCESS]{Colors.RESET} {Colors.GREEN}{msg}{Colors.RESET}"

