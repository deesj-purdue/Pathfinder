import pprint
from rplidarc1.scanner import RPLidar
import asyncio
import logging
import time

"""
This module provides an example of how to use the RPLidar class to perform
a simple scan and process the results.
"""


async def main(port: str, baud: int):
    """
    Main coroutine that demonstrates how to use the RPLidar class.

    This function creates three tasks:
    1. A task to wait for a specified time and then stop the scan
    2. A task to print the scan results as they are received
    3. A task to perform the scan itself

    After the scan is complete, it prints the final scan results and resets the device.

    Args:
        lidar (RPLidar): An initialized RPLidar instance.
    """
    # Create the RPLidar instance inside the running event loop so that
    # asyncio primitives (Event, Queue) are attached to the correct loop.
    lidar = RPLidar(port, baud)

    # Use create_task + gather for compatibility with Python < 3.11
    tasks = [
        asyncio.create_task(wait_and_stop(10, lidar.stop_event)),
        asyncio.create_task(queue_printer(lidar.output_queue, lidar.stop_event)),
        asyncio.create_task(lidar.simple_scan(make_return_dict=True)),
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception:
        # Ensure any remaining tasks are cancelled on error
        for t in tasks:
            if not t.done():
                t.cancel()
        # Wait for cancellations to finish
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        # Make sure all tasks are cleaned up
        await asyncio.gather(*[t for t in tasks], return_exceptions=True)

    if lidar.output_dict:
        pprint.pp(sorted(lidar.output_dict.items()))
    lidar.reset()


async def wait_and_stop(t, event: asyncio.Event):
    """
    Wait for a specified time and then set an event to stop the scan.

    Args:
        t (float): The time to wait in seconds.
        event (asyncio.Event): The event to set when the time has elapsed.
    """
    print("Start wait for end event")
    await asyncio.sleep(t)
    print("Setting stop event")
    event.set()


async def queue_printer(q: asyncio.Queue, event: asyncio.Event):
    """
    Print scan results from a queue until a stop event is set.

    This coroutine continuously reads scan results from the queue and prints them.
    If the queue has fewer than 10 items, it sleeps briefly to allow more data
    to accumulate.

    Args:
        q (asyncio.Queue): The queue containing scan results.
        event (asyncio.Event): An event that signals when to stop printing.
    """
    print("Start queue listener")
    while not event.is_set():
        if q.qsize() < 10:
            print("Printer sleeping for more data")
            await asyncio.sleep(0.1)
        out: dict = await q.get()
        print(out)


if __name__ == "__main__":
    logging.basicConfig(level=0)
    try:
        asyncio.run(main("COM5", 460800))
    except KeyboardInterrupt:
        pass
