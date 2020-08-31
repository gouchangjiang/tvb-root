# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
"""
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""

import os
import shutil
import threading
from queue import Queue
from threading import Lock

from syncrypto import Crypto, Syncrypto
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.core.decorators import synchronized

LOGGER = get_logger(__name__)

# Queue used to push projects which need synchronisation
sync_project_queue = Queue()

# Dict used to count the project frequency from the queue
queue_elements_count = {}

# Dict used to count how the project usage
users_project_usage = {}

# Dict used to keep projects which are marked for deletion
marked_for_delete = set()

lock = Lock()


def queue_count(folder):
    return queue_elements_count[folder] if folder in queue_elements_count else 0


def project_active_count(folder):
    return users_project_usage[folder] if folder in users_project_usage else 0


@synchronized(lock)
def inc_project_usage_count(folder):
    count = project_active_count(folder)
    count += 1
    users_project_usage[folder] = count


@synchronized(lock)
def dec_project_usage_count(folder):
    count = project_active_count(folder)
    if count == 1:
        users_project_usage.pop(folder)
        return
    count -= 1
    users_project_usage[folder] = count


@synchronized(lock)
def inc_queue_count(folder):
    count = queue_count(folder)
    count += 1
    queue_elements_count[folder] = count


@synchronized(lock)
def dec_queue_count(folder):
    count = queue_count(folder)
    if count == 1:
        queue_elements_count.pop(folder)
        return
    count -= 1
    queue_elements_count[folder] = count


@synchronized(lock)
def check_and_delete(folder):
    # Check if we can remove a folder:
    #   1. It is not in the queue
    #   2. It is marked for delete
    #   3. Nobody is using it
    if queue_count(folder) == 0 \
            and folder in marked_for_delete \
            and project_active_count(folder) == 0:
        marked_for_delete.remove(folder)
        shutil.rmtree(folder)


def set_project_inactive(project_folder):
    dec_project_usage_count(project_folder)
    if queue_count(project_folder) > 0 or project_active_count(project_folder) > 0:
        marked_for_delete.add(project_folder)
        LOGGER.info("Project {} still in use. Marked for deletion.".format(project_folder))
        return
    LOGGER.info("Remove project: {}".format(project_folder))
    shutil.rmtree(project_folder)


class DataEncryptionHandler:

    @staticmethod
    def compute_encrypted_folder_path(project_folder):
        return "{}_encrypted".format(project_folder)

    @staticmethod
    def sync_folders(folder):
        if not TvbProfile.current.web.ENCRYPT_STORAGE:
            return
        encrypted_folder = DataEncryptionHandler.compute_encrypted_folder_path(folder)
        # TODO: Fetch Password file
        crypto = Crypto("some_dummy_password")
        syncro = Syncrypto(crypto, encrypted_folder, folder)
        syncro.sync_folder()
        trash_path = os.path.join(encrypted_folder, "_syncrypto", "trash")
        if os.path.exists(trash_path):
            shutil.rmtree(trash_path)

    @staticmethod
    def set_project_active(project_folder):
        if not TvbProfile.current.web.ENCRYPT_STORAGE:
            return
        inc_project_usage_count(project_folder)
        DataEncryptionHandler.push_folder_to_sync(project_folder)

    @staticmethod
    def set_project_inactive(project_folder):
        if not TvbProfile.current.web.ENCRYPT_STORAGE:
            return
        set_project_inactive(project_folder)

    @staticmethod
    def push_folder_to_sync(project_folder):
        if not TvbProfile.current.web.ENCRYPT_STORAGE or queue_count(project_folder) > 2:
            return
        inc_queue_count(project_folder)
        sync_project_queue.put(project_folder)


class FoldersQueueConsumer(threading.Thread):
    was_processing = False

    marked_stop = False

    def mark_stop(self):
        self.marked_stop = True

    def run(self):
        if not TvbProfile.current.web.ENCRYPT_STORAGE:
            return
        while True:
            if sync_project_queue.empty():
                if self.was_processing:
                    self.was_processing = False
                    LOGGER.info("Finish processing queue")
                if self.marked_stop:
                    break
                continue
            if not self.was_processing:
                LOGGER.info("Start processing queue")
                self.was_processing = True
            folder = sync_project_queue.get()
            DataEncryptionHandler.sync_folders(folder)
            dec_queue_count(folder)
            check_and_delete(folder)
            sync_project_queue.task_done()