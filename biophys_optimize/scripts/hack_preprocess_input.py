#!/usr/bin/env python

import argparse
import numpy as np
import psycopg2
import json

CONNECTION_STRING = 'host=limsdb2 dbname=lims2 user=limsreader password=limsro'


def preprocess_info_from_lims(specimen_id):
    sql = """
        with dendrite_type as
        (
        select sts.specimen_id, st.name
        from specimen_tags_specimens sts
        join specimen_tags st on sts.specimen_tag_id = st.id
        where st.name like 'dendrite type%%'
        )
        select swc.storage_directory || swc.filename,
        nwb.storage_directory || nwb.filename,
        dt.name
        from specimens sp
        join dendrite_type dt on sp.id = dt.specimen_id
        join ephys_roi_results err on sp.ephys_roi_result_id = err.id
        join well_known_files nwb on nwb.attachable_id = err.id
        join neuron_reconstructions n on n.specimen_id = sp.id
        join well_known_files swc on swc.attachable_id = n.id
        where sp.id = %s
        and nwb.well_known_file_type_id = 475137571
        and nwb.attachable_type = 'EphysRoiResult'
        and n.manual and not n.superseded
        and swc.well_known_file_type_id = 303941301
        """

    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()
    cur.execute(sql, (specimen_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if not result:
        print "Could not find info for specimen ", specimen_id
        return None

    return result


def get_sweeps_of_type(sweep_type, specimen_id, passed_only=False):
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()
    sql = "SELECT sw.sweep_number FROM ephys_sweeps sw JOIN ephys_stimuli stim \
                 ON stim.id = sw.ephys_stimulus_id \
                 WHERE sw.specimen_id = %s AND stim.description LIKE %s"

    if passed_only:
        sql += "\nAND sw.workflow_state LIKE '%%passed'"

    cur.execute(sql, (specimen_id, '%' + sweep_type + '%'))
    sweeps = [s[0] for s in cur.fetchall()]
    cur.close()
    conn.close()

    return sweeps


def bridge_average(specimen_id, cap_check_sweeps):
    conn = psycopg2.connect(CONNECTION_STRING)
    cur = conn.cursor()

    sql = """select sw.bridge_balance_mohm from ephys_sweeps sw
         join ephys_stimuli stim on stim.id = sw.ephys_stimulus_id
         where sw.specimen_id = %s
         and sw.sweep_number in %s"""

    cur.execute(sql, (specimen_id, tuple(cap_check_sweeps)))

    sweeps_data = cur.fetchall()
    cur.close()
    conn.close()

    bridge_balances = [s[0] for s in sweeps_data]
    bridge_avg = np.mean(bridge_balances)

    return bridge_avg


def main():
    parser = argparse.ArgumentParser(description='Make a preprocess input json file')
    parser.add_argument('specimen_id')
    parser.add_argument('storage_directory')
    args = parser.parse_args()

    specimen_id = args.specimen_id
    storage_directory = args.storage_directory

    result = preprocess_info_from_lims(specimen_id)
    if not result:
        return
    swc_path, nwb_path, dendrite_type = result

    sweep_types = {
		"core_1_long_squares": "C1LSCOARSE",
		"core_2_long_squares": "C2SQRHELNG",
		"seed_1_noise": "C1NSSEED_1",
		"seed_2_noise": "C1NSSEED_2",
		"cap_checks": "C1SQCAPCHK",
	}

    sweeps = {k: get_sweeps_of_type(v, specimen_id, passed_only=True)
              for k, v in sweep_types.iteritems()}

    if len(sweeps["cap_checks"]) > 0:
        bridge_avg = bridge_average(specimen_id, sweeps["cap_checks"])
    else:
        bridge_avg = 0

    output = {
        "paths": {
            "nwb": nwb_path,
            "swc": swc_path,
            "storage_directory": storage_directory,
        },
        "dendrite_type_tag": dendrite_type,
        "sweeps": sweeps,
        "bridge_avg": bridge_avg,
    }

    with open(specimen_id + "_preprocess_input.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__": main()

