import os
import os.path as osp
import ssl
import urllib


def makedirs(path):
    os.makedirs(path, exist_ok=True)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2].split("?")[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print("Using exist file", filename)
        return path

    if log:
        print("Downloading", url)

    makedirs(folder)

    context = ssl._create_unverified_context()  # pylint: disable=protected-access
    with urllib.request.urlopen(url, context=context) as data:
        with open(path, "wb") as f:
            f.write(data.read())

    return path
