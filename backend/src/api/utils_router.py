from fastapi import APIRouter, HTTPException
import subprocess
import sys

router = APIRouter(prefix="/utils", tags=["utils"])

@router.post("/browse-file")
def browse_file():
    """
    Open a native file dialog on the server (Mac) to select a file.
    Returns the absolute path.
    """
    if sys.platform != "darwin":
        raise HTTPException(status_code=501, detail="File picker only supported on macOS for now")

    script = """
    tell application "System Events"
        activate
        set theFile to choose file with prompt "Select a measurement file"
        return POSIX path of theFile
    end tell
    """
    
    try:
        result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=True)
        path = result.stdout.strip()
        return {"path": path}
    except subprocess.CalledProcessError as e:
        # User likely cancelled
        return {"path": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
