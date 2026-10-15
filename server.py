import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(script_dir)

# NOTE: the heavy imports (server_manager pulls in torch and the whole model
# stack) live inside the __main__ guard on purpose.  On Windows the parallel
# video-decode workers use the "spawn" start method, which re-imports this
# module in every child process.  Keeping these imports out of module scope
# means those lightweight decode workers don't drag in torch / the model
# pipeline / the GitHub version check — they only need lib/.../mp_decode.py.
if __name__ == "__main__":
    import signal
    import asyncio
    from uvicorn import Config, Server
    from lib.server.server_manager import app, port
    import lib.server.routes  # noqa: F401  (registers the routes on `app` via decorators)
    from lib.utils.host_memory import HeapTrimmer

    _trimmer = HeapTrimmer()

    @app.middleware("http")
    async def trim_heap_after_work(request, call_next):
        response = await call_next(request)
        if request.method == "POST":
            # What a finished request freed in the heap goes back to the system.
            asyncio.get_running_loop().run_in_executor(None, _trimmer.maybe_trim)
        return response

    config = Config(app, host="0.0.0.0", port=port)
    server = Server(config)

    async def stop_servers():
        await server.shutdown()

    def signal_handler(s, f):
        asyncio.create_task(stop_servers())

    signal.signal(signal.SIGINT, signal_handler)

    server.run()