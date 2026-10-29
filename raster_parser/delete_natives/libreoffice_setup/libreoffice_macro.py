import os
import shutil
import threading
from pathlib import Path
from subprocess import run
import re
# import xml.etree.ElementTree as ET

# LibreOffice keeps its user profile under a profile-format version folder ("4" since
# LibreOffice 4.0).
LO_CONFIG_DIR = "~/.config/libreoffice/4"
# Same path the package __init__ used to pass when it registered on import.
MACRO_SCRIPT_FP = Path(__file__).parents[3] / "raster_parser/libreoffice_setup/DeleteDrawings.xba"

_registered = False
_register_lock = threading.Lock()


def ensure_macro_registered() -> None:
    """Register the DeleteDrawings macro once per process, before the first LibreOffice call."""
    global _registered
    with _register_lock:
        if not _registered:
            register_libreoffice_macro(LO_CONFIG_DIR, MACRO_SCRIPT_FP)
            _registered = True


def init_config():
    """Run to initialise default configuration for libreoffice."""
    cmd = [
        "libreoffice",
        "--invisible",
        "--headless",
        "--norestore",
        "--nologo",
        "--nofirststartwizard",
        "macro:///Standard.Module1.Main",
    ]
    run(cmd)


def register_libreoffice_macro(config_dir_path: str, script_fp: str | Path):
    """
    Safely injects the macro registration into script.xlb and updates
    registrymodifications.xcu using string manipulation to prevent LibreOffice
    from rejecting and factory-resetting the XML files.
    """

    # Initialise config dir
    init_config()

    # Resolve absolute paths
    base_dir = os.path.expanduser(config_dir_path)
    user_dir = os.path.join(base_dir, "user")
    standard_dir = os.path.join(user_dir, "basic", "Standard")

    script_xlb = os.path.join(standard_dir, "script.xlb")
    registry_xcu = os.path.join(user_dir, "registrymodifications.xcu")
    script_fp = Path(script_fp)
    module_name = script_fp.stem

    # ---------------------------------------------------------
    # 0. COPY XBA SCRIPT
    # ---------------------------------------------------------
    script_fn = script_fp.name
    try:
        shutil.copy(script_fp, os.path.join(standard_dir, script_fn))
    except OSError as exc:
        print(f"[!] Warning: could not copy macro into LibreOffice config: {exc}")
        return

    # ---------------------------------------------------------
    # 1. SAFELY UPDATE script.xlb (Registers the Module)
    # ---------------------------------------------------------
    if os.path.exists(script_xlb):
        with open(script_xlb, "r", encoding="utf-8") as f:
            xlb_content = f.read()

        # Check if it's already in the file
        if f'library:name="{module_name}"' not in xlb_content:
            # Inject our element exactly one line above the closing tag
            insertion = f' <library:element library:name="{module_name}"/>\n</library:library>'
            xlb_content = xlb_content.replace("</library:library>", insertion)

            with open(script_xlb, "w", encoding="utf-8") as f:
                f.write(xlb_content)
            print(f"[*] Success: Added '{module_name}' to script.xlb")
        else:
            print(f"[-] Notice: '{module_name}' is already registered in script.xlb")
    else:
        print(f"[!] Error: {script_xlb} not found. Is LibreOffice installed/run at least once?")

    # ---------------------------------------------------------
    # 2. SAFELY UPDATE registrymodifications.xcu (Macro Security)
    # ---------------------------------------------------------
    if os.path.exists(registry_xcu):
        with open(registry_xcu, "r", encoding="utf-8") as f:
            xcu_content = f.read()

        # The exact string LibreOffice expects to enable Medium/Low macro security (Level 1)
        # MacroSecurityLevel: 0 low, 1 medium, 2 high, 3 very high.
        security_item = '<oor:item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>1</value></prop></oor:item>'

        # Regex pattern to find the scripting security item if LibreOffice already generated it
        pattern = r'<oor:item oor:path="/org\.openoffice\.Office\.Common/Security/Scripting">.*?</oor:item>'

        if re.search(pattern, xcu_content, flags=re.DOTALL):
            # If it exists (e.g., set to high security), cleanly replace the whole block with ours
            new_xcu = re.sub(pattern, security_item, xcu_content, flags=re.DOTALL)

            if new_xcu != xcu_content:
                with open(registry_xcu, "w", encoding="utf-8") as f:
                    f.write(new_xcu)
                print(
                    "[*] Success: Updated existing Macro Security in registrymodifications.xcu to Level 1"
                )
            else:
                print(
                    "[-] Notice: Macro Security is already at Level 1 in registrymodifications.xcu"
                )
        else:
            # If it doesn't exist at all, inject it right before the end of the file
            new_xcu = xcu_content.replace("</oor:items>", f"{security_item}\n</oor:items>")
            with open(registry_xcu, "w", encoding="utf-8") as f:
                f.write(new_xcu)
            print("[*] Success: Appended Macro Security Level 1 to registrymodifications.xcu")
    else:
        print(f"[!] Error: {registry_xcu} not found.")
