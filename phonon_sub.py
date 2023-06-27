import json
import os
import random
import time
import numpy as np

import scc_lib
from poscar import Poscar

'''
run jobs, should run read_yaml first
to reduce the effect of pulay stress, the ENCUT should be increased

INCAR tags for NV- center

LHYPERFINE = .TRUE.
NGYROMAG = 307.66 1070.5
LEFG = .TRUE.
QUAD_EFG= 20.44 0
'''


def jobid_in_queue():
    bjobs_return = scc_lib.linux_command("bjobs")[1:]
    # there are jobs
    if bjobs_return:
        map_func = lambda x: list(filter(None, x.strip().split()))[:4:3]
        id_list, queue_list = np.array(list(map(map_func, bjobs_return))).T
        return id_list.tolist(), queue_list.tolist()
    # there isn't jobs
    else:
        return [], []


# the more powerful the queue is, the smaller the weight is
weight_for_each_queue = {"ckduan": 2, "smallopa": 1, "smallib": 2}


def recommend_queue(weight_for_each_queue, threhold):
    _, queue_list = jobid_in_queue()
    number_of_job = len(queue_list)
    while number_of_job > threhold:
        print(f"there are more than {threhold} jobs running, please wait ...")
        time.sleep(60)
        _, queue_list = jobid_in_queue()
        number_of_job = len(queue_list)
    recommend_dict = {queue: ((queue_list.count(queue) / number_of_job) if number_of_job > 0 else 0)
                             * weight_for_each_queue[queue]
                      for queue in weight_for_each_queue}

    argmin = np.array(list(recommend_dict.values())).argmin()
    return list(recommend_dict.keys())[argmin]


def get_vasp_state(_dir):
    id_list, _ = jobid_in_queue()

    # there is a flag already
    if os.path.isfile(f"{_dir}/vasp_done"):
        state = "vasp done"

    # the job has already been submitted
    elif os.path.isfile(f"{_dir}/jobid") and os.path.getsize(f"{_dir}/jobid") > 0:
        # get the jobid from jobid file
        with open(f"{_dir}/jobid", "r") as jobid:
            lines = jobid.readlines()

        # by default the first job is the vasp static calculation
        # and the second one is zfs calculation
        jobid_vasp_static = lines[-1].split("<")[1].split(">")[0]

        # there is vasp static log which means the job already finished
        if os.path.isfile(f"{_dir}/{jobid_vasp_static}.log"):
            with open(f"{_dir}/{jobid_vasp_static}.log", "r") as log:
                lines = log.readlines()
            # it depends on whether write wavecar
            if lines and lines[-1] and ('F=' in lines[-1] or 'writing wavefunctions' in lines[-1]):
                state = "vasp done"
                scc_lib.linux_command(f"touch {_dir}/vasp_done")
            else:
                state = "vasp bad end"

        # there is vasp static log which means the job already finished
        else:
            state = "vasp running" if jobid_vasp_static in id_list else "vasp bad end"

    # the job has not been submitted yet
    else:
        state = "vasp not submitted"
    print(f"path = {_dir}, vasp state = {state}")
    return state


def get_zfs_state(_dir, id_list):
    if os.path.isfile(f"{_dir}/zfs.json"):
        state = "zfs done"
    elif os.path.isfile(f"{_dir}/zfs_bad_end"):
        state = "zfs bad end"
    elif os.path.isfile(f"{_dir}/zfs_submitted"):
        # get the jobid from jobid file
        with open(f"{_dir}/jobid", "r") as jobid:
            lines = jobid.readlines()

        # by default the first job is the vasp static calculation
        # and the second one is zfs calculation
        jobid_zfs = lines[-1].split("<")[1].split(">")[0]
        if jobid_zfs in id_list:
            state = "zfs running"
        else:
            state = "zfs bad end"
            scc_lib.linux_command(f"touch {_dir}/zfs_bad_end")
    # not submit yet
    elif os.path.isfile(f"{_dir}/WAVECAR"):
        state = "zfs ready"
    else:
        state = "rerun"
    print(f"path = {_dir}, zfs state = {state}")
    return state


def post_processing(_prefix, zfs_calc):
    # get the running list
    id_list, _ = jobid_in_queue()

    for _dir in os.listdir():
        if _dir.split("_")[0] == _prefix and os.path.isdir(_dir):
            state = get_vasp_state(_dir, id_list)

            if state == "vasp not submitted":
                # resubmit
                scc_lib.linux_command(f"rm {_dir} -r")
            elif state == "vasp bad end":
                scc_lib.linux_command(f"rm {_dir} -r")
            elif state == "vasp done":
                # remove CHGCAR unconditionally
                for filename in ["CHG", "CHGCAR"]:
                    if os.path.isfile(f"{_dir}/{filename}"):
                        scc_lib.linux_command(f"rm {_dir}/{filename}")

                if zfs_calc:
                    state = get_zfs_state(_dir, id_list)
                    if state == "zfs done":
                        if os.path.isfile(f"{_dir}/WAVECAR"):
                            scc_lib.linux_command(f"rm {_dir}/WAVECAR")
                    elif state == "zfs running":
                        pass
                    elif state == "zfs bad end":
                        pass
                    elif state == "zfs ready":
                        queue = recommend_queue(weight_for_each_queue, 75)
                        scc_lib.linux_command(f"cd {_dir}; "
                                              f"situ_zfs {queue} {24 if queue == 'smallib' else 28}; "
                                              f"touch zfs_submitted; "
                                              f"cd ..;")
                    elif state == "rerun":
                        queue = recommend_queue(weight_for_each_queue, 75)
                        scc_lib.linux_command(f"cp INCAR WAVECAR {_dir}; "
                                              f"cd {_dir}; "
                                              f"rm vasp_done; "
                                              f"0run -q {queue}; "
                                              f"cd ..;")
                else:
                    scc_lib.linux_command(f"rm {_dir}/WAVECAR")


if __name__ == '__main__':
    zfs_calc = True
    C = 239.22657077980037
    json_path = "mode.json"
    prefix = "p"
    mass, frequency, motion = json.load(open(json_path, "r"))
    motion = np.array(motion)
    mass = np.array(mass)
    motion_without_mass = (motion.transpose([0, 2, 1]) / np.sqrt(mass)).transpose([0, 2, 1])
    number_of_mode = len(frequency)

    ground_state_path = "CONTCAR"
    ground_state_structure = Poscar(ground_state_path)
    ground_state_structure.d2c()

    post_processing(prefix, zfs_calc)

    for mode_index in range(number_of_mode)[3:]:

        '''
        Disk quotas for usr qif (uid 1292):
             Filesystem    used   quota   limit   grace   files   quota   limit   grace
         /home/phys/qif  776.7G   1000G   1000G       - 1284607       0       0       -
        Disk quotas for grp phys (gid 510):
             Filesystem    used   quota   limit   grace   files   quota   limit   grace
         /home/phys/qif      0k      0k      0k       -       0       0       0       -
        '''

        quota = list(filter(None, scc_lib.linux_command("lfs quota -h ~")[2].split()))[1]
        if float(quota.replace("G", "").replace("*", "")) > 950:
            post_processing(prefix, zfs_calc)

        for displace in [1, 1.5]:
            for sign in [1, -1]:
                # the motions are replaced by factors, since the motions are well normed.
                # print(frequency[mode_index], .5 * (displace * frequency[mode_index]) ** 2 * C)
                work_path = f"{prefix}_{mode_index}_{displace * sign}"
                if work_path not in os.listdir():
                    print(f"mode = {mode_index}, factor = {displace}, sign = {sign}")
                    queue = recommend_queue(weight_for_each_queue, 75)

                    a_motion = ground_state_structure.copy()
                    a_motion.d2c()
                    a_motion.table["position"] += motion_without_mass[mode_index] * displace * sign
                    a_motion.write(f"POSCAR_{mode_index}_{displace * sign}")
                    scc_lib.linux_command(f"mkdir {work_path}; "
                                          f"mv POSCAR_{mode_index}_{displace * sign} {work_path}/POSCAR; "
                                          f"cp INCAR POTCAR KPOINTS WAVECAR {work_path}; "
                                          f"cd {work_path}; "
                                          f"0run -q {queue}; "
                                          f"cd ..;")
