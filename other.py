import argparse
from visdom import Visdom


def get_default_visdom_env():
    """
    Create and return default environment from visdom
    Returns
    -------
    Visdom
           Visdom class object
    """
    default_port = 8097
    default_hostname = "http://localhost"
    parser = argparse.ArgumentParser(description='Demo arguments')
    parser.add_argument('-port', metavar='port', type=int, default=default_port,
                        help='port the visdom server is running on.')
    parser.add_argument('-server', metavar='server', type=str,
                        default=default_hostname,
                        help='Server address of the target to run the demo on.')
    flags = parser.parse_args()
    viz = Visdom(port=flags.port, server=flags.server)

    assert viz.check_connection(timeout_seconds=3), \
        'No connection could be formed quickly'
    return viz
