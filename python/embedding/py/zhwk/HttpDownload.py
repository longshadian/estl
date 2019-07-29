
import urllib.request
import os


def DownloadToFile(url, fdest):
    with urllib.request.urlopen(url=url) as f:
        with open(fdest, mode="w+b") as output:
            output.write(f.read(1024 * 16))


def Launch(url, download_path, fname):
    dest_path = os.path.join(download_path, fname)
    DownloadToFile(url=url, fdest=dest_path)




