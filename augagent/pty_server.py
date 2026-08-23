import asyncio
import sys
import os
from fastapi import WebSocket, WebSocketDisconnect

async def handle_terminal_ws(websocket: WebSocket):
    if sys.platform == "win32":
        await handle_win32_pty(websocket)
    else:
        await handle_unix_pty(websocket)

async def handle_win32_pty(websocket: WebSocket):
    try:
        from winpty import PtyProcess  # type: ignore
    except ImportError:
        await websocket.send_text("Error: pywinpty is not installed on this Windows system.\\r\\n")
        await websocket.close()
        return

    # Create a winpty process
    try:
        sandbox_mode = os.environ.get("AUGAGENT_SANDBOX", "false").lower() == "true"
        if sandbox_mode:
            proc = PtyProcess.spawn("docker run -it --rm ubuntu /bin/bash")
        else:
            # Launch cmd or powershell
            cmd = os.environ.get("COMSPEC", "cmd.exe")
            proc = PtyProcess.spawn(cmd)
    except Exception as e:
        await websocket.send_text(f"Error spawning PTY: {e}\\r\\n")
        await websocket.close()
        return
        
    async def read_from_pty():
        loop = asyncio.get_running_loop()
        while True:
            try:
                # Read from PTY
                data = await loop.run_in_executor(None, proc.read, 1024)
                if not data:
                    break
                await websocket.send_text(data)
            except EOFError:
                break
            except Exception as e:
                print(f"Error reading from PTY: {e}")
                break

    async def read_from_ws():
        try:
            while True:
                data = await websocket.receive_text()
                # Write to PTY
                proc.write(data)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Error reading from WS: {e}")

    read_task = asyncio.create_task(read_from_pty())
    write_task = asyncio.create_task(read_from_ws())

    done, pending = await asyncio.wait(
        [read_task, write_task], return_when=asyncio.FIRST_COMPLETED
    )

    for task in pending:
        task.cancel()

    try:
        proc.close()
    except:
        pass
    
    try:
        await websocket.close()
    except:
        pass

async def handle_unix_pty(websocket: WebSocket):
    try:
        import ptyprocess  # type: ignore
    except ImportError:
        await websocket.send_text("Error: ptyprocess is not installed.\\r\\n")
        await websocket.close()
        return

    try:
        sandbox_mode = os.environ.get("AUGAGENT_SANDBOX", "false").lower() == "true"
        if sandbox_mode:
            proc = ptyprocess.PtyProcessUnicode.spawn(["docker", "run", "-it", "--rm", "ubuntu", "/bin/bash"])
        else:
            shell = os.environ.get("SHELL", "/bin/bash")
            proc = ptyprocess.PtyProcessUnicode.spawn([shell])
    except Exception as e:
        await websocket.send_text(f"Error spawning PTY: {e}\\r\\n")
        await websocket.close()
        return

    async def read_from_pty():
        loop = asyncio.get_running_loop()
        while True:
            try:
                data = await loop.run_in_executor(None, proc.read, 1024)
                if not data:
                    break
                await websocket.send_text(data)
            except EOFError:
                break
            except Exception as e:
                print(f"Error reading from PTY: {e}")
                break

    async def read_from_ws():
        try:
            while True:
                data = await websocket.receive_text()
                proc.write(data)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Error reading from WS: {e}")

    read_task = asyncio.create_task(read_from_pty())
    write_task = asyncio.create_task(read_from_ws())

    done, pending = await asyncio.wait(
        [read_task, write_task], return_when=asyncio.FIRST_COMPLETED
    )

    for task in pending:
        task.cancel()

    try:
        proc.terminate(force=True)
    except:
        pass
        
    try:
        await websocket.close()
    except:
        pass
