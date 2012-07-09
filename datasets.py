""" Utilities to download NeuroImaging datasets
"""

import os
import urllib2
import tarfile
import zipfile
import gzip
import sys
import shutil
import time

import numpy as np
from sklearn.datasets.base import Bunch


def _chunk_report_(bytes_so_far, total_size, t0):
    """Show downloading percentage

    Parameters
    ----------
    bytes_so_far: integer
        Number of downloaded bytes

    total_size: integer, optional
        Total size of the file. None is valid

    t0: integer, optional
        The time in seconds (as returned by time.time()) at which the
        download was started.
    """
    if total_size:
        percent = float(bytes_so_far) / total_size
        percent = round(percent * 100, 2)
        dt = time.time() - t0
        # We use a max to avoid a division by zero
        remaining = (100. - percent) / max(0.01, percent) * dt
        sys.stderr.write(
            "Downloaded %d of %d bytes (%0.2f%%, %i seconds remaining)\r"
            % (bytes_so_far, total_size, percent, remaining))
    else:
        sys.stderr.write("Downloaded %d of ? bytes\r" % (bytes_so_far))


def _chunk_read_(response, local_file, chunk_size=8192, report_hook=None):
    """Download a file chunk by chunk and show advancement

    Parameters
    ----------
    response: urllib.addinfourl
        Response to the download request in order to get file size

    local_file: file
        Hard disk file where data should be written

    chunk_size: integer, optional
        Size of downloaded chunks. Default: 8192

    report_hook: boolean
        Whether or not to show downloading advancement. Default: None

    Returns
    -------
    data: string
        The downloaded file.

    """
    total_size = response.info().getheader('Content-Length').strip()
    try:
        total_size = int(total_size)
    except Exception, e:
        print "Total size could not be determined. Error: ", e
        total_size = None
    bytes_so_far = 0

    t0 = time.time()
    while 1:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)

        if not chunk:
            if report_hook:
                sys.stderr.write('\n')
            break

        local_file.write(chunk)
        if report_hook:
            _chunk_report_(bytes_so_far, total_size, t0)

    return


def _get_dataset_dir(dataset_name, data_dir=None):
    """Returns data directory of given dataset

    Parameters
    ----------
    dataset_name: string
        The unique name of the dataset.

    data_dir: string
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data_dir: string
        Path of the given dataset directory.

    """
    if not data_dir:
        data_dir = os.path.join(os.getcwd(), 'Data')
    data_dir = os.path.join(data_dir, dataset_name)
    return data_dir


def _uncompress_file(file_, delete_archive=True):
    """Uncompress files contained in a data_set.

    Parameters
    ----------
    file: string
        path of file to be uncompressed.

    delete_archive: boolean, optional
        Wheteher or not to delete archive once it is uncompressed.
        Default: True

    Notes
    -----
    This handles zip, tar, gzip and bzip files only.
    """
    print 'extracting data from %s...' % file_
    data_dir = os.path.dirname(file_)
    # We first try to see if it is a zip file
    try:
        if file_.endswith('.zip'):
            z = zipfile.Zipfile(file_)
            z.extractall(data_dir)
            z.close()
        elif file_.endswith('.gz'):
            z = gzip.GzipFile(file_)
            name = os.path.splitext(file_)[0]
            f = file(name, 'w')
            z = f.write(z.read())
        else:
            tar = tarfile.open(file_, "r")
            tar.extractall(path=data_dir)
            tar.close()
        if delete_archive:
            os.remove(file_)
            pass
        print '   ...done.'
    except Exception as e:
        print 'error: ', e
        raise


def _fetch_file(url, data_dir):
    """Load requested file, downloading it if needed or requested

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    urls: array of strings
        Contains the urls of files to be downloaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    files: array of string
        Absolute paths of downloaded files on disk

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded data are
    cleaned.
    """
    # Determine data path
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_name = os.path.basename(url)
    full_name = os.path.join(data_dir, file_name)
    if not os.path.exists(full_name):
        t0 = time.time()
        try:
            # Download data
            print 'Downloading data from %s ...' % url
            req = urllib2.Request(url)
            data = urllib2.urlopen(req)
            local_file = open(full_name, "wb")
            _chunk_read_(data, local_file, report_hook=True)
            dt = time.time() - t0
            print '...done. (%i seconds, %i min)' % (dt, dt / 60)
        except urllib2.HTTPError, e:
            print "HTTP Error:", e, url
            return None
        except urllib2.URLError, e:
            print "URL Error:", e, url
            return None
        finally:
            local_file.close()
    return full_name


def _fetch_dataset(dataset_name, urls, data_dir=None, uncompress=True):
    """Load requested dataset, downloading it if needed or requested

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    urls: array of strings
        Contains the urls of files to be downloaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    files: array of string
        Absolute paths of downloaded files on disk

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded data are
    cleaned.
    """
    # Determine data path
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    files = []
    for url in urls:
        full_name = _fetch_file(url, data_dir)
        if not full_name:
            print 'An error occured, abort fetching'
            shutil.rmtree(data_dir)
        if uncompress:
            try:
                _uncompress_file(full_name)
            except Exception:
                # We are giving it a second try, but won't try a third
                # time :)
                print 'archive corrupted, trying to download it again'
                _fetch_file(url, data_dir)
                _uncompress_file(full_name)
        files.append(os.path.splitext(full_name)[0])

    return files


def _get_dataset(dataset_name, file_names, data_dir=None):
    """Returns absolute paths of a dataset files if exist

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    file_names: array of strings
        File that compose the dataset to be retrieved on the disk.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    files: array of string
        List of dataset files on disk

    Notes
    -----
    If at least one file is missing, an IOError is raised.
    """
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)
    file_paths = []
    for file_name in file_names:
        full_name = os.path.join(data_dir, file_name)
        if not os.path.exists(full_name):
            raise IOError("No such file: '%s'" % full_name)
        file_paths.append(full_name)
    return file_paths


###############################################################################
# Dataset downloading functions

def fetch_data(data_dir=None):
    """Function returning the starplus data, downloading them if needed

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'datas' : a list of 6 numpy arrays representing the data to learn
        'targets' : list
                    targets of the data
        'masks' : the masks for the data. If indices is true, returns the
            coordinates of the voxels instead of a binary map to deal with
            sparse matrices.

    Notes
    -----
    Each element will be of the form :
    PATH/*.npy

    References
    ----------
    Documentation and data :
    http://noble.gs.washington.edu/yeast/
    """
    data_names = ['kernel_matrix_pfamdom_cn_3588',
                  'kernel_matrix_tap_n_3588',
                  'kernel_matrix_mpi_n_3588',
                  'kernel_matrix_mgi_n_3588',
                  'kernel_matrix_exp_diff_n_3588',
                  'kernel_matrix_exp_gauss_n_3588',
                  'kernel_matrix_pfamdom_exp_cn_3588',
                  'kernel_matrix_sw_cn_3588']
    dataset_files = [i + '.npy' for i in data_names]
    dataset_dir = _get_dataset_dir("", data_dir=None)

    try:
        _get_dataset("", dataset_files, data_dir=None)
    except IOError:
        file_names = [i + '.txt.gz' for i in data_names]
        url = 'http://noble.gs.washington.edu/yeast'

        urls = ["/".join([url, i]) for i in file_names]

        full_names = _fetch_dataset('', urls, data_dir=None)

        for index, full_name in enumerate(full_names):
            # Converting data to a more readable format
            print "Converting file %d on 8..." % (index + 1)
            # General information
            try:
                K = np.genfromtxt(full_name, skip_header=1)
                K = K[:, 1:]
                K = K.astype(np.float)

                name = dataset_files[index]
                name = os.path.join(dataset_dir, name)
                np.save(name, K)
                print "...done."

                # Removing the unused data
                os.remove(full_name)
            except Exception, e:
                print "Impossible to convert the file %s:\n %s " % (full_name, e)
                shutil.rmtree(dataset_dir)
                raise e

    print "...done."

    data = Bunch()
    for i in data_names:
        K = np.load(os.path.join(dataset_dir, i + ".npy"))
        data[i] = K

    return data
